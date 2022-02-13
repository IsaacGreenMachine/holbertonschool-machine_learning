def summation_i_squared(n):
    if n < 1:
        return None
    else:
        sum = 0
        for i in range(1, n+1):
            sum += (i)**2
        return sum
