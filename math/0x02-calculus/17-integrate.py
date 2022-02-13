def poly_integral(poly, C=0):
    if type(poly) is list and len(poly) > 0 and type(C) is int:
        new_list = []
        new_list.append(C)
        for i in range(1, len(poly)+1):
            new_list.append(poly[i - 1]/i)
        return new_list
    else:
        return None
