
import json
import os
import re

def print_course(subject, num, data):
    print(f"{subject} {num}: {data['Credits']} : {data['Name']}")
    if "Prerequisites" in data:
        print(f"\tPrepreqs : {data['Prerequisites']}")
    if "Corequisites" in data:
        print(f"\tCoreqs : {data['Corequisites']}")
    print('-'*30)


def offerings_in_quarter(quarter):
    subjects = ["BIO", "CHEM", "ENSS", "ENVS", "PHYS", "PHEV"]
    subjects = ["ACCT", "BUSN", "ENTP", "FIN", "MGMT", "TAX"]
    #subjects = ["CS", "MATH", "SE"]
    #subjects = ["PHIL"]
    #subjects = ["CULA","HRM"]
    for subject in subjects:
        f = open(f"Database\\Subjects\\{subject}.json")
        for num, data in json.load(f)["Courses"].items():
            if "Offerings" in data and "Summer Quarter 20-21" in data['Offerings']:
                print_course(subject, num, data)

def courses_by_credit_count(credit_count):
    for filename in os.listdir("Database\\Subjects"):
        f = open(f"Database\\Subjects\\{filename}")
        for num, data in json.load(f)["Courses"].items():
            if "Credits" in data and "Offerings" in data:
                try:
                    if float(data["Credits"]) == credit_count:
                        print_course(filename.split('.')[0], num, data)
                except:
                    pass

courses_by_credit_count(2.0)
