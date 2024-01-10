use anyhow::{bail, Context, Result};
use concurrent_queue::ConcurrentQueue;
use crossbeam::channel::Receiver;
use human_repr::HumanThroughput;
use itertools::Itertools;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    num::NonZeroUsize,
    path::Path,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

pub struct FileChunk {
    pub data: Vec<u8>,
    pub offset_from_start_of_file: u64,
}

/// A file reader that reads a file concurrently from multiple threads, where each thread reads one chunk of the file at a time.
/// Chunk order is implementation-defined. Only works for files with a well-known fixed size
pub struct ConcurrentFileReader {
    reader_thread: JoinHandle<Result<()>>,
}

impl ConcurrentFileReader {
    pub fn read<P: AsRef<Path>>(
        path: P,
        concurrency: NonZeroUsize,
        chunk_size: NonZeroUsize,
        start_offset_in_file: usize,
    ) -> (Self, Receiver<FileChunk>) {
        let path = path.as_ref().to_path_buf();
        let concurrency = concurrency.get();
        let chunk_size = chunk_size.get();

        let (sender, receiver) = crossbeam::channel::unbounded::<FileChunk>();

        let handle = std::thread::spawn(move || -> Result<()> {
            let path = &path;
            let file_metadata = path.metadata()?;
            let num_chunks = (file_metadata.len() as usize + chunk_size - 1) / chunk_size;
            let actual_concurrency = concurrency.min(num_chunks);
            let num_chunks_per_thread = (num_chunks + actual_concurrency - 1) / actual_concurrency;

            std::thread::scope(|s| {
                let handles = (0..actual_concurrency)
                    .map(|thread_id| {
                        let sender = &sender;
                        s.spawn(move || -> Result<()> {
                            let mut reader = File::open(path)?;
                            let first_chunk = thread_id * num_chunks_per_thread;
                            let last_chunk_exclusive =
                                ((thread_id + 1) * num_chunks_per_thread).min(num_chunks);
                            let first_byte = first_chunk * chunk_size + start_offset_in_file;
                            reader.seek(SeekFrom::Start(first_byte as u64))?;
                            for chunk_id in first_chunk..last_chunk_exclusive {
                                let mut buf: Vec<u8> = vec![0; chunk_size];
                                let num_bytes_read = reader.read(&mut buf)?;
                                if num_bytes_read < chunk_size {
                                    buf.resize(num_bytes_read, 0);
                                }
                                let chunk_start_byte = chunk_id * chunk_size;
                                sender.send(FileChunk {
                                    data: buf,
                                    offset_from_start_of_file: chunk_start_byte as u64,
                                })?;
                            }

                            Ok(())
                        })
                    })
                    .collect::<Vec<_>>();
                handles.into_iter().try_for_each(|h| -> Result<()> {
                    match h.join() {
                        Ok(r) => r,
                        Err(e) => bail!("Read thread paniced with payload {e:?}"),
                    }
                })
            })
        });

        (
            Self {
                reader_thread: handle,
            },
            receiver,
        )
    }

    pub fn join(self) -> Result<()> {
        match self.reader_thread.join() {
            Ok(r) => r,
            Err(e) => bail!("Reader thread paniced with payload {e:?}"),
        }
    }
}

pub struct AtomicF64 {
    storage: AtomicU64,
}

impl AtomicF64 {
    pub fn new(value: f64) -> Self {
        let bits = value.to_bits();
        Self {
            storage: AtomicU64::new(bits),
        }
    }
    pub fn store(&self, value: f64, ordering: Ordering) {
        let as_u64 = value.to_bits();
        self.storage.store(as_u64, ordering)
    }
    pub fn load(&self, ordering: Ordering) -> f64 {
        let as_u64 = self.storage.load(ordering);
        f64::from_bits(as_u64)
    }
}

pub struct ThroughputCounter {
    /// The last `window_size` measured throughput values, in work items per second
    last_throughputs: Vec<f64>,
    /// How many throughput values to save at the most? Determines the size of the
    /// window for computing a running average of the throughput
    window_size: usize,
}

impl ThroughputCounter {
    pub(crate) const DEFAULT_CAPACITY: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(16) };

    pub fn new(window_size: NonZeroUsize) -> Self {
        Self {
            last_throughputs: Vec::with_capacity(window_size.get()),
            window_size: window_size.get(),
        }
    }

    pub fn add_measurement(&mut self, throughput: f64) {
        self.last_throughputs.push(throughput);
        if self.last_throughputs.len() > self.window_size {
            self.last_throughputs.remove(0);
        }
    }

    pub fn get_average(&self) -> f64 {
        if self.last_throughputs.is_empty() {
            0.0
        } else {
            self.last_throughputs.iter().copied().sum::<f64>() / self.last_throughputs.len() as f64
        }
    }
}

pub struct PipelineWorkItem {
    /// The function to execute
    func: Box<dyn FnMut() -> Result<Vec<Self>> + Send>,
    /// What type of work item is this?
    type_id: usize,
    /// How many elements of work does this work item process? This is an abstract quantity and depends on
    /// the type of work item. It can be things like 'number of points' or 'bytes'
    work_count: usize,
}

impl PipelineWorkItem {
    pub fn new<F: FnMut() -> Result<Vec<Self>> + Send + 'static>(
        type_id: usize,
        work_count: usize,
        func: F,
    ) -> Self {
        Self {
            func: Box::new(func),
            type_id,
            work_count,
        }
    }
}

pub struct WorkQueueStatistics {
    upper_bound_remaining_work: usize,
    estimated_throughput: ThroughputCounter,
}

pub struct WorkQueue {
    queue: ConcurrentQueue<PipelineWorkItem>,
    statistics: Mutex<WorkQueueStatistics>,
    estimated_remaining_runtime: AtomicF64,
}

impl WorkQueue {
    pub(crate) fn new(upper_bound_remaining_work: usize) -> Self {
        Self {
            queue: ConcurrentQueue::unbounded(),
            statistics: Mutex::new(WorkQueueStatistics {
                upper_bound_remaining_work,
                estimated_throughput: ThroughputCounter::new(ThroughputCounter::DEFAULT_CAPACITY),
            }),
            estimated_remaining_runtime: AtomicF64::new(f64::MAX),
        }
    }

    pub(crate) fn update_statistics(&self, finished_work_count: usize, runtime: Duration) {
        let mut statistics = self.statistics.lock().expect("Lock was poisoned");
        let throughput = finished_work_count as f64 / runtime.as_secs_f64();
        statistics.estimated_throughput.add_measurement(throughput);
        if finished_work_count > statistics.upper_bound_remaining_work {
            panic!("Can't finish more work than the upper bound of remaining work");
        }
        statistics.upper_bound_remaining_work -= finished_work_count;
        let new_throughput = statistics.estimated_throughput.get_average();
        let new_estimated_runtime = statistics.upper_bound_remaining_work as f64 / new_throughput;
        self.estimated_remaining_runtime
            .store(new_estimated_runtime, Ordering::SeqCst);
    }

    pub(crate) fn reset(&self) {
        let mut statistics = self.statistics.lock().expect("Lock was poisoned");
        statistics.estimated_throughput =
            ThroughputCounter::new(ThroughputCounter::DEFAULT_CAPACITY);
        self.estimated_remaining_runtime
            .store(f64::MAX, Ordering::SeqCst);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum WakeReason {
    NewWorkItem,
    Terminate,
}

pub struct SyncState {
    num_idle_threads: AtomicUsize,
    thread_went_idle_sender: crossbeam::channel::Sender<()>,
    waker: Condvar,
    wake_reason: Mutex<WakeReason>,
}

impl SyncState {
    pub(crate) fn reset(&self) {
        // self.num_idle_threads.store(0, Ordering::SeqCst);
        let mut wake_reason = self.wake_reason.lock().expect("Lock was poisoned");
        *wake_reason = WakeReason::NewWorkItem;
    }
}

pub struct ConcurrentPipeline {
    work_queues: Arc<HashMap<usize, WorkQueue>>,
    threads: Vec<JoinHandle<Result<()>>>,
    sync_state: Arc<SyncState>,
    thread_went_idle_receiver: crossbeam::channel::Receiver<()>,
}

impl ConcurrentPipeline {
    pub fn new<I: IntoIterator<Item = (usize, usize)>>(
        work_item_types: I,
        num_threads: usize,
    ) -> Self {
        let mut work_queues: HashMap<usize, WorkQueue> = Default::default();
        for (work_item_type, upper_bound_remaining_work) in work_item_types {
            if work_queues
                .insert(work_item_type, WorkQueue::new(upper_bound_remaining_work))
                .is_some()
            {
                panic!("Work item type {} must be unique!", work_item_type);
            }
        }
        let work_queues = Arc::new(work_queues);

        let (thread_went_to_sleep_sender, thread_went_to_sleep_receiver) =
            crossbeam::channel::bounded(num_threads);

        let sync_state = Arc::new(SyncState {
            num_idle_threads: AtomicUsize::new(0),
            thread_went_idle_sender: thread_went_to_sleep_sender,
            wake_reason: Mutex::new(WakeReason::NewWorkItem),
            waker: Condvar::new(),
        });

        Self {
            work_queues: work_queues.clone(),
            threads: (0..num_threads)
                .map(|_| {
                    let work_queues = work_queues.clone();
                    let sync_state = sync_state.clone();
                    std::thread::spawn(move || Self::worker_thread(work_queues, sync_state))
                })
                .collect(),
            sync_state,
            thread_went_idle_receiver: thread_went_to_sleep_receiver,
        }
    }

    /// Runs a pipeline based on the given initial work items to completion. Blocks until all work items
    /// have been processed
    pub fn run(
        &self,
        initial_work_items: impl IntoIterator<Item = PipelineWorkItem>,
    ) -> Result<()> {
        self.sync_state.reset();
        self.work_queues
            .values()
            .for_each(|work_queue| work_queue.reset());

        // Enqueue all initial work items, this should wake up a bunch of worker threads
        let mut had_at_least_one_work_item = false;
        for work_item in initial_work_items {
            had_at_least_one_work_item = true;
            Self::enqueue_work_item(work_item, &self.work_queues, &self.sync_state);
        }

        if !had_at_least_one_work_item {
            return Ok(());
        }

        // Wait until all work items have been processed
        loop {
            match self.thread_went_idle_receiver.recv() {
                Ok(_) => {
                    // Check if all threads are idle
                    let expected_idle_count = self.threads.len();
                    let actual_idle_count = self.sync_state.num_idle_threads.load(Ordering::SeqCst);
                    tracy_client::plot!("num_idle_threads", actual_idle_count as f64);
                    if expected_idle_count == actual_idle_count {
                        // eprintln!("Pipeline run is done");
                        break;
                    }
                }
                Err(why) => bail!("Error while trying to receive thread idle notification: {why}"),
            }
        }

        // Print statistics
        for (task_id, work_queue) in self.work_queues.as_ref() {
            let stats = work_queue.statistics.lock().expect("Lock was poisoned");
            let throughput = stats.estimated_throughput.get_average();
            eprintln!(
                "Est. throughput for task type {}: {:.2}",
                task_id,
                throughput.human_throughput("items"),
            );
        }

        Ok(())
    }

    pub fn enqueue_work_item(
        work_item: PipelineWorkItem,
        work_queues: &HashMap<usize, WorkQueue>,
        sync_state: &SyncState,
    ) {
        let matching_queue = work_queues
            .get(&work_item.type_id)
            .expect("Invalid type_id of work item, no matching work queue found");
        matching_queue.queue.push(work_item);
        sync_state.waker.notify_one();
    }

    fn worker_thread(
        work_queues: Arc<HashMap<usize, WorkQueue>>,
        sync_state: Arc<SyncState>,
    ) -> Result<()> {
        // TODO What happens with work items that return an error? How does this affect the whole pipeline?

        // TODO How do the threads signal that they are done? I guess whenever one thread wakes, goes through
        // all queues and finds no work items, this might be a candidate for being done. If all threads do
        // this, then we are indeed done. So we have to keep some kind of counter?

        let try_pull_work_item = || -> Option<PipelineWorkItem> {
            let queues_by_highest_remaining_time = work_queues.values().sorted_by(|a, b| {
                let runtime_b = b.estimated_remaining_runtime.load(Ordering::SeqCst);
                let runtime_a = a.estimated_remaining_runtime.load(Ordering::SeqCst);
                if runtime_b < runtime_a {
                    std::cmp::Ordering::Less
                } else if runtime_b > runtime_a {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            queues_by_highest_remaining_time
                .filter_map(|queue| queue.queue.pop().ok())
                .next()
        };

        loop {
            // Pull work items as long as there are any
            while let Some(mut next_work_item) = try_pull_work_item() {
                let _span = tracy_client::span!("process work item");
                // Process the work item and update its associated queue
                let timer = Instant::now();
                let new_work_items =
                    (next_work_item.func)().context("Error while processing work item")?;
                let work_item_runtime = timer.elapsed();

                let work_item_queue = work_queues.get(&next_work_item.type_id).unwrap();
                work_item_queue.update_statistics(next_work_item.work_count, work_item_runtime);

                // Enqueue new work items, if there are any
                for work_item in new_work_items {
                    Self::enqueue_work_item(work_item, &work_queues, &sync_state);
                }
            }

            // Once we have no more work items, signal that we are idle
            sync_state.num_idle_threads.fetch_add(1, Ordering::SeqCst);
            sync_state.thread_went_idle_sender.send(())?;

            // eprintln!("Worker thread going idle");
            // wait until enqueue_work_item wakes this thread
            // TODO Wait in every loop iteration, or only wait if we can't pull any more work items?
            let wake_reason = sync_state
                .wake_reason
                .lock()
                .and_then(|lock| sync_state.waker.wait(lock));
            let lock = match wake_reason {
                Ok(lock) => {
                    if *lock == WakeReason::Terminate {
                        break;
                    }
                    lock
                }
                Err(poison) => bail!("Waker mutex was poisioned: {poison}"),
            };
            // eprintln!("Worker thread woke");

            // If this thread was idle, mark it as not idle anymore and decrease the number of idle threads
            sync_state.num_idle_threads.fetch_sub(1, Ordering::SeqCst);

            // TODO When to drop the lock?
            drop(lock);
        }

        Ok(())
    }
}

impl Drop for ConcurrentPipeline {
    fn drop(&mut self) {
        let mut wake_reason = self
            .sync_state
            .wake_reason
            .lock()
            .expect("Lock was poisoned");
        *wake_reason = WakeReason::Terminate;
        self.sync_state.waker.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use rand::{distributions::Standard, thread_rng, Rng};
    use scopeguard::defer;

    use super::*;

    #[test]
    fn test_concurrent_file_reader_is_correct() -> Result<()> {
        let rng = thread_rng();
        const SIZE: usize = 2 * 1024 * 1024;

        let chunk_sizes = vec![
            1024,
            1025,
            4096,
            12345,
            128 * 1024,
            SIZE - 1,
            SIZE,
            SIZE + 1,
        ];
        let concurrencies = vec![1, 2, 3, 4, 8, 16, 31, 32];

        let expected_data = rng
            .sample_iter::<u8, _>(Standard)
            .take(SIZE)
            .collect::<Vec<_>>();

        let path = "test_concurrent_file_reader_is_correct.bin";
        std::fs::write(path, &expected_data)?;
        defer! {
            std::fs::remove_file(path).expect("Failed to remove temporary file");
        }

        for chunk_size in &chunk_sizes {
            for concurrency in &concurrencies {
                let (reader, receiver) = ConcurrentFileReader::read(
                    path,
                    NonZeroUsize::new(*concurrency).unwrap(),
                    NonZeroUsize::new(*chunk_size).unwrap(),
                    0,
                );

                let mut actual_data: Vec<u8> = vec![0; SIZE];
                while let Ok(chunk) = receiver.recv() {
                    let chunk_range_in_file = chunk.offset_from_start_of_file as usize
                        ..(chunk.offset_from_start_of_file as usize + chunk.data.len());
                    actual_data[chunk_range_in_file].copy_from_slice(&chunk.data);
                }

                reader.join()?;

                assert_eq!(expected_data, actual_data, "ConcurrentFileReader yields wrong results for file size {SIZE}, chunk size {chunk_size} and concurrency factor {concurrency}");
            }
        }

        Ok(())
    }
}
