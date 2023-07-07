#Returns list of n by n elements of iterable
def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)

#Transform initial string data into list of 3D body points
def str2float(s):
    s = s.replace('[', '')
    s = s.replace(']', '')
    list = s.split(', ')
    list = [float(i) for i in list]

    new_list = []
    for x, y, z, ci in grouped(list, 4):
        new_list.append([x, y, z, ci])
    return new_list