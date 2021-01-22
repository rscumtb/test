my_str = "Hello"

test_iter = iter(my_str)
while True:
    try:
        each = next(test_iter)
    except StopIteration:
        break
    print(each)


while True:
    try:
        each = next(test_iter)
    except StopIteration:
        break
    print(each)
