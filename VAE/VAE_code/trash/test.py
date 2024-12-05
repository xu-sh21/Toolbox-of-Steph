def function_a(x):
    return x * 2

def function_b(func, y):
    return func(y)

test_func = function_a
# 将 function_a 作为参数传递给 function_b
result = function_b(test_func, 5)

print(result)  # 输出 10
