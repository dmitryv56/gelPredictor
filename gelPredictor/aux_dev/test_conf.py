#!/usr/bin/env python3

from configparser import ConfigParser
print("A!")
def main():
    pass
if __name__ =="__main__":
    print("B!")
    parser = ConfigParser()
    parser.read("/home/dmitry/LaLaguna/gelPredictor/aux_dev/test_config.ini")
    print("C!")
    print(parser.__str__)
    for section_name in parser.sections():
        print("Section: {}".format(section_name))
        print("Options: {}".format(parser.options(section_name)))
        for name, value in parser.items(section_name):
            print("{}: {}".format(name,value))

        pass
    pass
    main()
