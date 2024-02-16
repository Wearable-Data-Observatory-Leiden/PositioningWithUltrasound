from itertools import permutations

numbers = [1, 2, 3, 4, 5, 6]

def adjust_combination(combination):
    adjusted_combination = []
    for num in combination:
        adjusted_num = num - (combination[0] - 1)
        if adjusted_num <= 0:
            adjusted_num += 6  # Loop around to the end of the range
        adjusted_combination.append(adjusted_num)
    return adjusted_combination

# Generate combinations of 3 numbers
combinations_list = list(permutations(numbers, 3))

# List to store original combinations that meet the condition
original_combinations = []

# Perform the adjustment for each combination, check for ascending, and store the original combinations
for combination in combinations_list:
    adjusted_combination = adjust_combination(combination)
    # Check if adjusted combination is ascending
    is_ascending = all(adjusted_combination[i] < adjusted_combination[i + 1] for i in range(len(adjusted_combination) - 1))
    if is_ascending:
        print(f"Original: {combination}, Adjusted: {adjusted_combination}")
        original_combinations.append(combination)

# Print original combinations that meet the condition
print("Original combinations that meet the condition:")
for combination in original_combinations:
    print(combination)

hack_combination = [
(1, 2, 3),(1, 2, 4),(1, 2, 5),(1, 2, 6),(1, 3, 4),(1, 3, 5),(1, 3, 6),(1, 4, 5),(1, 4, 6),(1, 5, 6),
(2, 3, 1),(2, 3, 4),(2, 3, 5),(2, 3, 6),(2, 4, 1),(2, 4, 5),(2, 4, 6),(2, 5, 1),(2, 5, 6),(2, 6, 1),
(3, 1, 2),(3, 4, 1),(3, 4, 2),(3, 4, 5),(3, 4, 6),(3, 5, 1),(3, 5, 2),(3, 5, 6),(3, 6, 1),(3, 6, 2),
(4, 1, 2),(4, 1, 3),(4, 2, 3),(4, 5, 1),(4, 5, 2),(4, 5, 3),(4, 5, 6),(4, 6, 1),(4, 6, 2),(4, 6, 3),
(5, 1, 2),(5, 1, 3),(5, 1, 4),(5, 2, 3),(5, 2, 4),(5, 3, 4),(5, 6, 1),(5, 6, 2),(5, 6, 3),(5, 6, 4),
(6, 1, 2),(6, 1, 3),(6, 1, 4),(6, 1, 5),(6, 2, 3),(6, 2, 4),(6, 2, 5),(6, 3, 4),(6, 3, 5),(6, 4, 5)
]
for entry in hack_combination:
    print(entry)