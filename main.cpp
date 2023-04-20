#include <cstddef>
#include <cassert>
#include <memory>
#include <limits>

#include <cstring>
#include <vector>

#include <iostream>

//#define NDEBUG
#ifndef NDEBUG
#define DEBUG
#pragma message ("debug mode on")
#endif

static constexpr std::size_t power2(uint32_t exponent)
{
    return ((std::size_t) 1) << exponent;
}

static uint32_t compute_buddy_offset(uint32_t offset, uint32_t block_size)
{
    return offset ^ block_size;
}

static uint32_t compute_smaller_buddy_offset(uint32_t offset, uint32_t block_size)
{
    return offset & ~block_size;
}

template <typename FlagType>
void set_flag(FlagType & value, FlagType mask)
{
    value |= mask;
}

template <typename FlagType>
void unset_flag(FlagType & value, FlagType mask)
{
    value &= ~mask;
}

template <typename FlagType>
bool test_flag(FlagType value, FlagType mask)
{
    return (value & mask) == mask;
}

static uint32_t const BLOCK_HEADER_FLAG_FREE = 1;

struct block_user_data
{
    std::byte data[2];
};

struct block_header
{
    // flags are mostly unused and may be used for other things
    uint32_t flags;
    // This may not need as many bits and can be reused.
    uint16_t bucket_index;
    // 8 byte alignment that serves as additional user data
    block_user_data user_data;

    bool is_free()
    {
        return test_flag(flags, BLOCK_HEADER_FLAG_FREE);
    }

    void set_free()
    {
        set_flag(flags, BLOCK_HEADER_FLAG_FREE);
    }

    void unset_free()
    {
        unset_flag(flags, BLOCK_HEADER_FLAG_FREE);
    }
};

struct free_block_header : block_header
{
    std::size_t next_offset;
    std::size_t prev_offset;
};

// Does not work for n == 0
static constexpr uint32_t log2_ceil(std::size_t n, uint32_t max_bits_in_n)
{
    // Round upwards to the next power by subtracting one before and
    // adding one after (implicitly).
    uint32_t leading_zeroes = __builtin_clzl(n - 1);
    return max_bits_in_n - leading_zeroes;
}

class buddy_memory_pool
{
    static constexpr uint32_t MAXIMUM_EXPONENT = sizeof(std::size_t) * 8 - 1;
    static constexpr uint32_t MINIMUM_EXPONENT = log2_ceil(sizeof(free_block_header), sizeof(std::size_t) * 8);
    static constexpr uint32_t const NUM_BUCKETS = MAXIMUM_EXPONENT - MINIMUM_EXPONENT + 1;

    static constexpr std::size_t NO_BLOCK = std::numeric_limits<std::size_t>::max();

    public:

    static constexpr std::size_t MINIMUM_BLOCK_SIZE = power2(MINIMUM_EXPONENT);
    static constexpr std::size_t MAXIMUM_BLOCK_SIZE = power2(MAXIMUM_EXPONENT);

    buddy_memory_pool(std::byte * storage_ptr, std::size_t size_exponent)
        : storage_ptr_(storage_ptr)
        , size_exponent_(size_exponent)
    {
        uint32_t const free_block_index = size_exponent - MINIMUM_EXPONENT;
        for (uint32_t block_index = 0; block_index < free_block_index; block_index++)
        {
            free_lists_[block_index] = NO_BLOCK;
        }
        initialize_empty_bucket(free_block_index, 0);
        for (uint32_t block_index = free_block_index + 1; block_index < NUM_BUCKETS; block_index++)
        {
            free_lists_[block_index] = NO_BLOCK;
        }
    }

    // Allocates a block that uses internally the exact amount of memory but
    // provides a little less.
    std::byte * allocate_at_most(std::size_t n, std::size_t & actual_size)
    {
        actual_size = n - sizeof(block_header);
        return allocate(actual_size);
    }

    std::byte * allocate(std::size_t n)
    {
        std::size_t unused;
        return allocate(n, unused);
    }

    std::byte * allocate(std::size_t n, std::size_t & actual_size)
    {
        uint32_t bucket_index = compute_bucket_index(n + sizeof(block_header));

        std::size_t offset = pop_bucket(bucket_index);

        if (offset == NO_BLOCK)
        {
            // Try to split bigger block.

            uint32_t bigger_bucket_index = bucket_index;

            do
            {
                bigger_bucket_index++;
                if (bigger_bucket_index >= NUM_BUCKETS)
                {
                    // We are out of memory.
                    return nullptr;
                }
            }
            while (bucket_empty(bigger_bucket_index));

            // Update the free list where we found the block.
            std::size_t const bigger_block_offset = pop_non_empty_bucket(bigger_bucket_index);

            std::size_t current_block_size = block_size_from_bucket_index(bigger_bucket_index - 1);

            // Continuously split blocks until we reached our size.
            for (uint32_t current_bucket_index = bigger_bucket_index; current_bucket_index != bucket_index; current_bucket_index--, current_block_size /= 2)
            {
                // Update remaining free lists by writing the second block to the smaller free list.
                std::size_t smaller_block_offset = bigger_block_offset + current_block_size;

                // Write the second block to the smaller free list.
                initialize_empty_bucket(current_bucket_index - 1, smaller_block_offset);
            }

            // The beginning of the original biggest block is now our allocated block.
            offset = bigger_block_offset;

            block_header * header = header_pointer_at(offset);
            header->unset_free();
            header->bucket_index = bucket_index;
            actual_size = current_block_size - sizeof(block_header);
        }
        else
        {
            header_pointer_at(offset)->unset_free();
            actual_size = block_size_from_bucket_index(bucket_index);
        }
#ifdef DEBUG
        std::cout << "Buddy-allocating block at offset " << offset << std::endl;
#endif
        return pointer_at(offset + sizeof(block_header));
    }

    void free(std::byte * p)
    {
        std::size_t const freed_offset = block_offset_from_pointer(p);
        uint32_t bucket_index = header_pointer_at(freed_offset)->bucket_index;

        insert_promote_linked_list(bucket_index, block_size_from_bucket_index(bucket_index), freed_offset);
    }

    std::size_t count_free_memory()
    {
        std::size_t block_size = MINIMUM_BLOCK_SIZE;
        std::size_t result = 0;
        for (uint32_t bucket_index = 0; bucket_index < NUM_BUCKETS; bucket_index++, block_size *= 2)
        {
            std::size_t block_offset = free_lists_[bucket_index];

            while (block_offset != NO_BLOCK)
            {
                result += block_size;
                block_offset = free_header_pointer_at(block_offset)->next_offset;
            }
        }
        return result;
    }

    std::size_t allocation_size(std::byte * p)
    {
        uint32_t size_exponent = header_pointer_at(block_offset_from_pointer(p))->bucket_index + MINIMUM_EXPONENT;
        return power2(size_exponent) - sizeof(block_header);
    }

    block_user_data & get_allocator_user_data(std::byte * p)
    {
        return header_pointer_at(block_offset_from_pointer(p))->user_data;
    }

    bool is_managed(std::byte * p)
    {
        return storage_ptr_ < p && p < storage_ptr_ + power2(size_exponent_);
    }

    std::byte * align_pointer_to_block_size(std::byte * p, uint32_t factor_exponent)
    {
        std::ptrdiff_t offset = std::abs(storage_ptr_ - p) - sizeof(block_header);
        std::ptrdiff_t rem = offset % (MINIMUM_BLOCK_SIZE << factor_exponent);
        std::ptrdiff_t new_offset = offset - rem;
        return storage_ptr_ + new_offset + sizeof(block_header);
    }

    void visualize_memory(std::byte * p)
    {
        std::size_t const offset = block_offset_from_pointer(p);
        std::cout << "Address " << p << std::endl;
        std::cout << "Offset " << offset << std::endl;
    }

    void visualize_free_lists()
    {
        for (uint32_t bucket_index = 0; bucket_index < NUM_BUCKETS; bucket_index++)
        {
            std::size_t current_offset = free_lists_[bucket_index];
            if (current_offset != NO_BLOCK)
            {
                std::cout << "Free list for size " << block_size_from_bucket_index(bucket_index) << ": ";

                do
                {
                    free_block_header * free_header = free_header_pointer_at(current_offset);

                    std::cout << current_offset << ' ';
                    if (!free_header->is_free())
                    {
                        std::cout << " (not marked as free!)" << std::endl;
                    }
                    if (free_header->bucket_index != bucket_index)
                    {
                        std::cout << " (bucket index is wrong: " << free_header->bucket_index << ")" << std::endl;
                    }

                    std::cout << std::endl;
                    current_offset = free_header->next_offset;
                }
                while (current_offset != NO_BLOCK);
                std::cout << std::endl;
            }
        }
    }

    private:

    uint32_t compute_bucket_index(std::size_t n)
    {
        if (n < MINIMUM_BLOCK_SIZE)
        {
            return 0;
        }
        else
        {
            return log2_ceil(n, sizeof(std::size_t) * 8) - MINIMUM_EXPONENT;
        }
    }

    std::size_t block_size_from_bucket_index(uint32_t bucket_index)
    {
        return power2(bucket_index + MINIMUM_EXPONENT);
    }

    std::size_t block_offset_from_pointer(std::byte * p)
    {
        return p - storage_ptr_ - sizeof(block_header);
    }

    bool bucket_empty(uint32_t bucket_index)
    {
        return free_lists_[bucket_index] == NO_BLOCK;
    }

    void pop_offset(uint32_t bucket_index, uint32_t block_offset)
    {
        free_block_header * header = free_header_pointer_at(block_offset);
        std::size_t const free_block_offset = header->next_offset;
        free_lists_[bucket_index] = free_block_offset;
        header->unset_free();
    }

    std::size_t pop_non_empty_bucket(uint32_t bucket_index)
    {
        std::size_t const block_offset = free_lists_[bucket_index];

        pop_offset(bucket_index, block_offset);

        return block_offset;
    }

    std::size_t pop_bucket(uint32_t bucket_index)
    {
        std::size_t const block_offset = free_lists_[bucket_index];
        if (block_offset != NO_BLOCK)
        {
            pop_offset(bucket_index, block_offset);
        }
        return block_offset;
    }

    void initialize_empty_bucket(uint32_t bucket_index, std::size_t block_offset)
    {
        free_block_header * header = free_header_pointer_at(block_offset);
        header->next_offset = NO_BLOCK;
        header->set_free();
        header->bucket_index = bucket_index;
        free_lists_[bucket_index] = block_offset;
    }

    void insert_promote_linked_list(uint32_t bucket_index, std::size_t block_size, std::size_t freed_offset)
    {
        if (bucket_empty(bucket_index))
        {
            initialize_empty_bucket(bucket_index, freed_offset);
        }
        else
        {
            std::size_t anchor_offset = free_lists_[bucket_index];
            std::size_t buddy_offset = compute_buddy_offset(freed_offset, block_size);

            block_header * buddy_header = header_pointer_at(buddy_offset);

            // Try to merge.
            if (buddy_header->is_free() && buddy_header->bucket_index == bucket_index)
            {
                free_block_header * free_buddy_header = reinterpret_cast<free_block_header *>(buddy_header);
                if (buddy_offset == anchor_offset)
                {
                    // Special case for list anchor
                    free_lists_[bucket_index] = free_header_pointer_at(buddy_offset)->next_offset;
                }
                else
                {
                    // Unlink
                    free_block_header * next_block_header = free_header_pointer_at(free_buddy_header->next_offset);
                    free_block_header * prev_block_header = free_header_pointer_at(free_buddy_header->prev_offset);
                    prev_block_header->next_offset = free_buddy_header->next_offset;
                    next_block_header->prev_offset = free_buddy_header->prev_offset;
                }
                insert_promote_linked_list(bucket_index + 1, block_size * 2, compute_smaller_buddy_offset(freed_offset, block_size));
            }
            else
            {
                free_block_header * freed_header = free_header_pointer_at(freed_offset);
                free_block_header * anchor_header = free_header_pointer_at(anchor_offset);
                freed_header->next_offset = anchor_offset;
                freed_header->set_free();
                freed_header->bucket_index = bucket_index;
                anchor_header->prev_offset = freed_offset;
                free_lists_[bucket_index] = freed_offset;

                // prev_offset is not used
                //freed_header->prev_offset = NO_BLOCK;
            }
        }
    }

    free_block_header * free_header_pointer_at(std::size_t offset)
    {
        return typed_pointer_at<free_block_header>(offset);
    }

    block_header * header_pointer_at(std::size_t offset)
    {
        return typed_pointer_at<block_header>(offset);
    }

    template <typename T>
    T * typed_pointer_at(std::size_t offset)
    {
        return reinterpret_cast<T *>(pointer_at(offset));
    }

    std::byte * pointer_at(std::size_t offset)
    {
        return storage_ptr_ + offset;
    }

    std::size_t offset_from_pointer(std::byte * pointer)
    {
        // ptrdiff_t will always fit size_t.
        return storage_ptr_ - pointer;
    }

    std::byte * storage_ptr_;

    std::size_t size_exponent_;

    // Offset to the first free block of the bucket size.
    std::size_t free_lists_[NUM_BUCKETS];
};

void purge(buddy_memory_pool & m, std::byte * p)
{
    if (m.is_managed(p))
    {
        std::size_t n = m.allocation_size(p);
        std::memset(p, 0, n);
    }
}

static uint16_t const SLAB_HEADER_FLAG_SMALL_ALLOCATION = 1;

struct slab_block_header
{
    uint16_t flags;

    void set_small_allocation()
    {
        set_flag(flags, SLAB_HEADER_FLAG_SMALL_ALLOCATION);
    }

    void unset_small_allocation()
    {
        unset_flag(flags, SLAB_HEADER_FLAG_SMALL_ALLOCATION);
    }

    bool is_small_allocation()
    {
        return test_flag(flags, SLAB_HEADER_FLAG_SMALL_ALLOCATION);
    }
};

// Set the lowest zero bit and return the index in a single 64 bit chunk.
uint32_t bitset_get_and_set_single(uint64_t & bitset)
{
    uint64_t zero_bit_set = (bitset + 1) & -bitset;

    bitset |= zero_bit_set;


    if (bitset == 0)
    {
        bitset |= 1;
        return 0;
    }
    else
    {
        return 63 - __builtin_clzl(zero_bit_set);
    }
}

/**
 * Set the lowest zero bit and return the index in a series of contiguous 64
 * bit chunks.  The bitset cannot be full.
 */
uint32_t bitset_get_and_set_lowest_zero(uint64_t * start, uint32_t num_chunks)
{
    uint64_t const * const end = start + num_chunks;
    constexpr uint64_t FULL_BITSET_BLOCK = std::numeric_limits<uint64_t>::max();

    uint32_t index = 0;
    uint64_t * current = start;
    do
    {
        if (*current != FULL_BITSET_BLOCK)
        {
            return index + bitset_get_and_set_single(*current);
        }
        current++;
        index += 64;
    }
    while (current != end);

    // Bitset is full, should not happen
    assert(false);
}

/**
 * Set the bit at an index to zero in a series of contiguous 64 bit chunks.
 * The index has to be smaller than the bitset size.
 */
void bitset_unset(uint64_t * start, uint32_t index)
{
    uint32_t n = index / 64;
    uint32_t r = index % 64;
    start[n] &= ~(((uint64_t) 1) << r);
}

/**
 * Test if no bit is set in a series of contiguous 64 bit chunks.
 */
bool bitset_is_empty(uint64_t const * start, uint32_t num_chunks)
{
    uint64_t const * const end = start + num_chunks;
    uint64_t const * current = start;

    while (current != end)
    {
        if (*current != 0)
        {
            return false;
        }
        current++;
    }
    return true;
}

/**
 * Clear all bits in a series of contiguous 64 bit chunks.
 */
void bitset_clear(uint64_t * start, uint32_t num_chunks)
{
    uint64_t const * const end = start + num_chunks;
    while (start != end)
    {
        *start = 0;
        start++;
    }
}

/**
 * Clear all set bits except for the first one in a series of contiguous 64 bit
 * chunks.
 */
void bitset_clear_and_set_first(uint64_t * start, uint32_t num_chunks)
{
    *start = 1;
    bitset_clear(start + 1, num_chunks - 1);
}


struct slab_header
{
    slab_header * next;
    slab_header * prev;
    uint16_t element_size;
    uint16_t free_bitset_num_chunks;
    uint32_t capacity;
    // This field grows on demand.
    uint64_t free_bitset;

    bool is_empty() const
    {
        return bitset_is_empty(&free_bitset, free_bitset_num_chunks);
    }

    uint32_t allocate()
    {
        return bitset_get_and_set_lowest_zero(&free_bitset, free_bitset_num_chunks);
    }

    void free(uint32_t offset)
    {
        bitset_unset(&free_bitset, offset);
    }

    void unlink()
    {
        slab_header * next = this->next;
        slab_header * prev = this->prev;

        prev->next = next;
        next->prev = prev;
    }
};

struct slab_pool_manager
{
    static constexpr std::size_t NUM_SLABS = buddy_memory_pool::MINIMUM_BLOCK_SIZE - 2;

    /**
     * @param slab_factor_exponent Specifies at which point slab allocation
     * will happen.  The minimum block size of the underlying allocator is
     * multiplied by the given power of two.  Allocations below that size will
     * be allocated as slabs that share their block header.  Allocations at or
     * above that size will be allocated with the allocator.
     */
    slab_pool_manager(buddy_memory_pool & allocator, int slab_factor_exponent)
        : allocator_(allocator)
        , slab_factor_exponent_(slab_factor_exponent)
    {

        for (uint32_t i = 0; i < NUM_SLABS; i++)
        {
            slab_headers_[i] = nullptr;
            full_slab_headers_[i] = nullptr;
        }
    }

    std::size_t slab_size()
    {
        return buddy_memory_pool::MINIMUM_BLOCK_SIZE << slab_factor_exponent_;
    }

    std::byte * allocate(std::size_t n)
    {
        if (n + sizeof(block_header) < slab_size())
        {
            // do slab allocation
            return slab_allocate(n);
        }
        else
        {
            std::byte * p = allocator_.allocate(n);
            if (p == nullptr)
            {
                return nullptr;
            }
            block_header_from_allocation_pointer(p).unset_small_allocation();
            return p;
        }
    }

    void free(std::byte * p)
    {
        if (p != nullptr)
        {
            std::byte * allocation_pointer = allocation_pointer_from_element_pointer(p);
            if (block_header_from_allocation_pointer(allocation_pointer).is_small_allocation())
            {
                slab_free(allocation_pointer, p);
            }
            else
            {
                allocator_.free(p);
            }
        }
    }

    void visualize_memory(std::byte * p)
    {

    }

    private:

    slab_header & slab_header_from_allocation_pointer(std::byte * p)
    {
        // The slab_header is at the beginning of the usable memory of an
        // allocaiton.
        return *reinterpret_cast<slab_header *>(p);
    }

    std::byte * allocation_pointer_from_slab_header(slab_header & header)
    {
        return reinterpret_cast<std::byte *>(&header);
    }

    slab_block_header & block_header_from_allocation_pointer(std::byte * p)
    {
        // The 16 bit of user data for the allocator is used to store the
        // slab_block_header.
        block_user_data & user_data = allocator_.get_allocator_user_data(p);
        return *reinterpret_cast<slab_block_header *>(&user_data.data);
    }

    uint32_t compute_number_of_bitset_blocks(std::size_t element_size, std::size_t available_memory)
    {
        // This is the result of the following linear equation system.
        //
        // avail_memory = bitset + capacity * element_size
        // bitset = capacity / 64
        // capacity = (avail_memory - bitset) / element_size

        // At least 1 bitset block is required.
        return ((available_memory - 8) / (1 + 64 * element_size)) + 1;
    }

    std::byte * element_pointer_from_allocation_pointer
        ( std::byte * p
        , uint8_t free_bitset_num_chunks
        , std::size_t element_size
        , uint32_t element_index
        )
    {
        return p + slab_first_element_offset(free_bitset_num_chunks) + element_size * element_index;
    }

    std::byte * allocation_pointer_from_element_pointer(std::byte * p)
    {
        return allocator_.align_pointer_to_block_size(p, slab_factor_exponent_);
    }

    std::byte * slab_allocate(std::size_t n)
    {
        slab_header * header = slab_headers_[n];

#ifdef DEBUG
        std::cout << "Slab-allocating element of size " << n << std::endl;
#endif
        if (header == nullptr)
        {
            // Make sure the allocation fits exactly.
            std::size_t allocation_size;
            std::byte * p = allocator_.allocate_at_most(slab_size(), allocation_size);

            if (p == nullptr)
            {
                return nullptr;
            }
            else
            {
                std::size_t usable_size = allocation_size - sizeof(slab_header) - sizeof(uint64_t);
#ifdef DEBUG
                std::cout << "Allocating slab at " << p << " with size " << allocation_size << std::endl;
#endif

                slab_header & header = slab_header_from_allocation_pointer(p);

#ifdef DEBUG
                std::cout << "Found slab header at " << &header << std::endl;
#endif

                slab_block_header & block_header = block_header_from_allocation_pointer(p);
                block_header.set_small_allocation();
                
                uint32_t free_bitset_num_chunks = compute_number_of_bitset_blocks(n, usable_size);
                header.free_bitset_num_chunks = free_bitset_num_chunks;
                uint32_t capacity = (usable_size - free_bitset_num_chunks * sizeof(uint64_t)) / n;
                header.next = nullptr;
                header.prev = nullptr;
                header.element_size = n;
                // TODO The offsets of smaller elements will probably not be
                // big enough but depending on slab_factor_exponent_ it may be.
                // Use upper limit.
                header.capacity = capacity;

                bitset_clear_and_set_first(&header.free_bitset, free_bitset_num_chunks);

                link_header(n, &header);
                std::byte * result = element_pointer_from_allocation_pointer(p, free_bitset_num_chunks, n, 0);
#ifdef DEBUG
                std::cout << "Result address is " << result << std::endl << std::endl;
#endif
                return result;
            }
        }
        else
        {
#ifdef DEBUG
            std::cout << "Found existing slab at " << header << std::endl;
#endif
            uint32_t index = header->allocate();

#ifdef DEBUG
            std::cout << "Allocated memory in slab at index " << index << std::endl;
#endif

            if (index + 1 == header->capacity)
            {
#ifdef DEBUG
                std::cout << "Header full, moving to full headers" << std::endl;
#endif

                // move to full ones
                unlink_header(n, header);
                link_full_header(n, header);
            }

            std::byte * allocation_pointer = allocation_pointer_from_slab_header(*header);
            std::byte * result =
                element_pointer_from_allocation_pointer
                    ( allocation_pointer
                    , header->free_bitset_num_chunks
                    , n
                    , index
                    );
#ifdef DEBUG
            std::cout << "Result address is " << result << std::endl << std::endl;
#endif
            return result;
        }
    }

    uint32_t slab_first_element_offset(std::byte * allocation_pointer)
    {
        slab_header & header = slab_header_from_allocation_pointer(allocation_pointer);
        return slab_first_element_offset(header.free_bitset_num_chunks);
    }

    uint32_t slab_first_element_offset(uint32_t free_bitset_num_chunks)
    {
        return sizeof(slab_header) - sizeof(uint64_t) + free_bitset_num_chunks * sizeof(uint64_t);
    }

    void slab_free(std::byte * allocation_pointer, std::byte * element_pointer)
    {
#ifdef DEBUG
        std::cout << "Freeing " << element_pointer << " for slab block " << allocation_pointer << std::endl;
#endif
        slab_header & header = slab_header_from_allocation_pointer(allocation_pointer);

        std::ptrdiff_t element_offset = std::abs(element_pointer - allocation_pointer);

        uint32_t index = (element_offset - slab_first_element_offset(allocation_pointer)) / header.element_size;

#ifdef DEBUG
        std::cout << "Found offset " << element_offset << ", index " << index << ", element_size " << header.element_size << std::endl;
#endif

        header.free(index);

        if (header.is_empty())
        {
#ifdef DEBUG
            std::cout << "Slab is empty, freeing" << std::endl;
#endif
            unlink_header(header.element_size, &header);
            allocator_.free(allocation_pointer);
        }
        else if (index + 1 == header.capacity)
        {
            // Move back from full to normal slabs.
            unlink_full_header(header.element_size, &header);
            link_header(header.element_size, &header);
        }
    }

    void link_header_generic(int n, slab_header ** headers, slab_header * header)
    {
        slab_header * next = headers[n];

        header->next = next;

        headers[n] = header;
    }

    void unlink_header_generic(int n, slab_header **headers, slab_header * header)
    {
        if (headers[n] == header)
        {
            headers[n] = header->next;
        }
        else
        {
            header->unlink();
        }
    }

    void link_header(int n, slab_header * header)
    {
        link_header_generic(n, slab_headers_, header);
    }

    void unlink_header(int n, slab_header * header)
    {
        unlink_header_generic(n, slab_headers_, header);
    }

    void link_full_header(int n, slab_header * header)
    {
        link_header_generic(n, full_slab_headers_, header);
    }

    void unlink_full_header(int n, slab_header * header)
    {
        unlink_header_generic(n, full_slab_headers_, header);
    }

    buddy_memory_pool & allocator_;
    uint32_t slab_factor_exponent_;

    // Slabs for every size in ]0, MINIMUM_BLOCK_SIZE[.
    slab_header * slab_headers_[NUM_SLABS];

    slab_header * full_slab_headers_[NUM_SLABS];

};

int main()
{
    uint32_t const exponent = 10;
    uint32_t const size = power2(exponent);
    std::byte memory[size];
    for (unsigned int i = 0; i < size; i++)
    {
        memory[i] = (std::byte) size;
    }

    buddy_memory_pool m = buddy_memory_pool(memory, exponent);

    std::byte * p1 = m.allocate(2);

    std::byte * p2 = m.allocate(4);
    std::byte * p4 = m.allocate(4);
    std::byte * p5 = m.allocate(4);
    std::byte * p6 = m.allocate(4);

    std::cout << "Free memory: " << m.count_free_memory() << std::endl;

    std::byte * p3 = m.allocate(150);

    purge(m, p1);
    purge(m, p2);
    purge(m, p3);
    purge(m, p4);
    purge(m, p5);
    purge(m, p6);

    m.visualize_memory(p1);
    m.visualize_memory(p2);
    m.visualize_memory(p3);
    m.visualize_memory(p4);
    m.visualize_memory(p5);
    m.visualize_memory(p6);


    std::cout << "Free memory: " << m.count_free_memory() << std::endl;

    m.free(p3);

    std::cout << "Free memory: " << m.count_free_memory() << std::endl;

    m.free(p1);
    m.free(p2);
    m.free(p4);
    m.free(p5);
    m.free(p6);

    m.visualize_free_lists();

    std::cout << "Pool size overhead: " << sizeof(buddy_memory_pool) << std::endl;

    std::cout << "Free memory: " << m.count_free_memory() << std::endl;

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "slab_block_header size = " << sizeof(slab_block_header) << std::endl;
    std::cout << "slab_header size = " << sizeof(slab_header) << std::endl;

    slab_pool_manager m2 = slab_pool_manager(m, 4);

    std::vector<std::byte *> ps;
    for (int i = 0; i < 100; i++)
    {
        std::cout << "Allocation #" << i << std::endl;
        ps.push_back(m2.allocate(4));
    }
    std::byte * p15 = m2.allocate(5);


    int i = 99;
    while (!ps.empty())
    {
        std::cout << "Free #" << i << std::endl;
        m2.free(ps.back());
        ps.pop_back();
        i--;
    }
    m2.free(p15);

}
