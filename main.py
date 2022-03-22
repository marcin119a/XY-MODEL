from XY_model import XYSystem

if __name__ == '__main__':
    xy_system_1 = XYSystem(temperature=0.1, width=16)
    xy_system_2 = XYSystem(temperature=0.5, width=16)
    xy_system_3 = XYSystem(temperature=1, width=16)

    xy_system_1.equilibrate()
    xy_system_2.equilibrate()
    xy_system_3.equilibrate()
    xy_system_1.show_map()
    xy_system_2.show_map()
    xy_system_3.show_map()

