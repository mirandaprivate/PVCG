py_binary(
    name = 'compute_acceptance',
    srcs = ['compute_acceptance.py'],
)

py_test(
    name = 'compute_acceptance_test',
    size = 'small',
    srcs = ['compute_acceptance_test.py'],
    data = ['test_data'],
    deps = [':compute_acceptance'],
)

py_binary(
    name = 'generate_x_train',
    srcs = ['generate_x_train.py'],
    data = ['x_test.txt', 'x_train.txt'],
)

py_binary(
    name = 'targetfunction',
    srcs = ['targetfunction.py'],
    data = ['x_train.txt', 'x_test.txt'],
)