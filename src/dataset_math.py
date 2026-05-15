def generate_expressions(num_samples=None):

    expressions = [

        # Higher degree rationals
        "(x**3 - 1)/(x - 1)",
        "(x**3 - 8)/(x - 2)",
        "(x**4 - 16)/(x**2 - 4)",
        "(x**3 - x)/(x)",
        "(x**4 - 1)/(x**2 - 1)",

        # Nested algebra
        "(x**2 - 4)/(x - 2) + (x - 2)",
        "(x**2 - 9)/(x - 3) - 1",
        "(x**2 - 1)/(x - 1) + (x - 1)",
        "(x**2 - 4)/(x - 2) + sin(x)**2 + cos(x)**2",
        "(x**2 - 16)/(x - 4) + (x - 4)",

        # Trig complexity
        "sin(x)**4 + 2*sin(x)**2*cos(x)**2 + cos(x)**4",
        "1 - sin(x)**2",
        "(1 - cos(2*x))/2",
        "tan(x)**2 + 1",
        "sin(x)**2 - cos(x)**2",

        # Log rules
        "log(a*b*c)",
        "log(a**2 * b) - log(a)",
        "log(a/b) + log(b)",
        "log(a**3) - 3*log(a)",
        "log((a*b)/c)",

        # Mixed harder
        "(x**3 - 1)/(x - 1) + (x - 1)",
        "(x**2 - 4)/(x - 2) + (x**2 - 9)/(x - 3)",
        "(x**4 - 16)/(x - 2)",
        "(x**3 - 27)/(x - 3)",
        "(x**2 - 1)/(x - 1) + sin(x)**2",

        # Multi-variable
        "(a**2 - b**2)/(a - b)",
        "(a**3 - b**3)/(a - b)",
        "a**m * a**n / a**p",
        "(a*b)**2 / a",
        "log(a*b) - log(b)"
    ]

    return expressions