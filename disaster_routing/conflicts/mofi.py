def calc_mofi(num_fses: list[int], sol: list[int]) -> int:
    return max(a + b for a, b in zip(sol, num_fses))
