#!/usr/bin/env python3
"""
Factorial Calculator

This script provides functions to calculate the factorial of a number using both
iterative and recursive approaches. It includes error handling and a demonstration
of the functions through user interaction.
"""


def factorial_iterative(n):
    """
    Calculate factorial using an iterative approach.
    
    Args:
        n (int): A non-negative integer
        
    Returns:
        int: The factorial of n
        
    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer
    """
    # Check if n is an integer
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    
    # Check if n is non-negative
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    # Base case: 0! = 1
    if n == 0:
        return 1
    
    # Iterative calculation
    result = 1
    for i in range(1, n + 1):
        result *= i
    
    return result


def factorial_recursive(n):
    """
    Calculate factorial using a recursive approach.
    
    Args:
        n (int): A non-negative integer
        
    Returns:
        int: The factorial of n
        
    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer
    """
    # Check if n is an integer
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    
    # Check if n is non-negative
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    # Base case: 0! = 1
    if n == 0:
        return 1
    
    # Recursive case: n! = n * (n-1)!
    return n * factorial_recursive(n - 1)


def get_valid_input():
    """
    Get and validate user input for factorial calculation.
    
    Returns:
        int: A valid non-negative integer
    """
    while True:
        try:
            user_input = input("Enter a non-negative integer: ")
            number = int(user_input)
            
            if number < 0:
                print("Error: Please enter a non-negative integer.")
                continue
                
            return number
        
        except ValueError:
            print("Error: Please enter a valid integer.")


def main():
    """
    Main function to demonstrate factorial calculation.
    """
    print("Factorial Calculator")
    print("====================")
    
    try:
        # Get user input
        number = get_valid_input()
        
        # Calculate factorial using both methods
        iterative_result = factorial_iterative(number)
        recursive_result = factorial_recursive(number)
        
        # Display results
        print(f"\nFactorial of {number}:")
        print(f"Using iterative approach: {iterative_result}")
        print(f"Using recursive approach: {recursive_result}")
        
        # Verify both methods produce the same result
        if iterative_result == recursive_result:
            print("\nBoth methods produced the same result.")
        else:
            print("\nWarning: Methods produced different results!")
            
    except RecursionError:
        # Handle recursion depth exceeded (for large numbers)
        print("\nError: Maximum recursion depth exceeded. The number is too large for the recursive approach.")
        print("Try using only the iterative approach for large numbers.")
    
    except Exception as e:
        # Handle any other unexpected errors
        print(f"\nAn unexpected error occurred: {e}")


# Execute main function when script is run directly
if __name__ == "__main__":
    main()
