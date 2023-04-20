This projects implements two strategies for allocation as a proof of concept.  Some of the decisions probably do not make that much sense but it was just an experiment.

# Buddy Allocator

A buddy allocator manages memory with the buddy allocation algorithm.  The advantage is very fast constant time allocation.  However, sizes are always powers of two and that way up to half of the space of an allocation may be wasted.

# Slab Allocator

A slab allocator allocates bigger chunks of memory and then allocates single slabs of those tiles to provide smaller allocation sizes.

* Slabs are allocated from another allocator.
* The size and a bitset of the slab are located in the beginning of the slab.  This is only possible in constant time if the size of slabs is limited, and that is the case.
* To compute the address of a slab from an allocation pointer in constant time, a fixed block size has been used.
* Allocations with sizes above some threshold are allocated with the original allocator.  This is not optimal.  Perhaps, there is a scheme one can use to get slabs of different sizes and still get to the slab in constant time.
