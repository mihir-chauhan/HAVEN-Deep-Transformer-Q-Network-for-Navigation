def double_hashing(keys, p=4969):
    hash_table = {}
    def h(k):
        return k % p
    def g(k):
        return (k + 1) % (p - 2)
    for key in keys:
        key_str = str(key)
        if len(key_str) < 9:
            key_str = key_str.zfill(9)  # Pad with leading zeros to make 9 digits
        k = int(key_str)
        pos = h(k)
        step = g(k)
        i = 0
        while pos in hash_table:
            i += 1
            pos = (h(k) + i * step) % p
        hash_table[pos] = k
    return hash_table

employee_ssns = [
    132489971,
    509496993,
    546332190,
    '034367980',  # String to preserve leading zero
    '047900151',  # String to preserve leading zero
    329938157,
    212228844,
    325510778,
    353354519,
    '053708912'   # String to preserve leading zero
]

# Convert all to integers (handling leading zeros)
processed_ssns = []
for ssn in employee_ssns:
    if isinstance(ssn, str):
        # Remove leading zeros for proper integer conversion
        processed_ssns.append(int(ssn))
    else:
        processed_ssns.append(ssn)

# Perform double hashing
result = double_hashing(processed_ssns)

# Print the results in a nice table format
print("Employee SSN Assignment:")
print("{:<12} {:<10}".format("SSN", "Memory Location"))
print("-" * 25)
for pos, ssn in sorted(result.items(), key=lambda x: x[0]):
    # Format SSN with leading zeros if necessary
    formatted_ssn = str(ssn).zfill(9)
    print("{:<12} {:<10}".format(formatted_ssn, pos))

# Print the hash table load factor
load_factor = len(result) / 4969
print(f"\nHash table load factor: {load_factor:.4f}")