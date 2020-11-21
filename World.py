class World():



    def __init__(self):

        origins = [(-5, 0), (-10, 0), (-15, 0)]

        for origin in origins:
            env = Environment(origin)

            