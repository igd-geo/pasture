# pasture

Rust library for point cloud processing

**Under development** 

## Core

- [ ] PointBuffer
    - [x] InterleavedPointBuffer
    - [x] PerAttributePointBuffer
        - [ ] Improve `push_attribute` and `push_attribute_range`
- [ ] Point Views
    - [x] Interleaved view
    - [x] PerAttribute view
    - [ ] Performance checks on views for different PointBuffers
- [ ] PointType
    - [x] Basic `PointType` structure
    - [ ] Procedural macro for implementing `PointType` for a type

## I/O

- [ ] LAS/LAZ
    - [ ] Reader
        - [x] Format 0
        - [x] Format 1
        - [x] Format 2
        - [x] Format 3
        - [x] Format 4
        - [x] Format 5
        - [ ] Format 6
        - [ ] Format 7
        - [ ] Format 8
        - [ ] Format 9
        - [ ] Format 10
        - [ ] Attribute conversions (e.g. positions as I32, F32, F64)
        - [ ] SeekToPoint
            - [ ] Tests
    - [ ] Writer
        - [x] Format 0
        - [x] Format 1
        - [x] Format 2
        - [x] Format 3
        - [x] Format 4
        - [x] Format 5
        - [ ] Format 6
        - [ ] Format 7
        - [ ] Format 8
        - [ ] Format 9
        - [ ] Format 10
        - [ ] Attribute conversions (e.g. positions as I32, F32, F64)
        - [ ] SeekToPoint
            - [ ] Support SeekToPoint in Writer? 
    - [ ] Metadata
        - [x] Basic metadata structure
        - [ ] Support for additional attributes in header
        - [ ] Support for VLRs
            - [ ] Define how VLRs should be represented 
- [ ] 3D Tiles
    - [ ] Reader
    - [ ] Writer