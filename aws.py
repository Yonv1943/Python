# Airways Management System using MySql Connector

'''
Make Sure you have installed mysql.connector modules
- Python My SQL Connector - pip install mysql-connector
'''

import mysql.connector


sqlpswd = input("Enter SQL USER Password : ")

''' Since the User name of most SQL users are same i.e. 'root' '''
flightDB = mysql.connector.connect(host='localhost', user='root', passwd=sqlpswd)

# Fetching the Databases in our Program
cursor = flightDB.cursor()

# Will Create Database Flight If it's not created before
try:
    cursor.execute('CREATE DATABASE flight')
except:
    pass

# Using Database flight
cursor.execute('USE flight')

# Creating Table of Flight's Data
try:
    cursor.execute('CREATE TABLE flightData(flightNo INT, fName VARCHAR(20), fare INT)')
except:
    pass

line = "----------------------------------------------------------------"
def write_data():
    ''' It will add New Details of Flights in table flightData'''
    No_Of_flight = int(input(f"\n{line}\nEnter the Number of Flights you want to insert : "))
    for num in range(No_Of_flight):
        print(f'{line}\nEnter the Details of {num+1}-Flight : \n')
        flightNo = int(input(f'Enter flight Number : '))
        fName = input(f'Enter Name of flight-{flightNo} : ')
        fare = input(f'Enter Fare of {fName}-{flightNo} : ')
        cursor.execute(f"INSERT INTO flightData VALUES ({flightNo}, '{fName}', {fare})")
        flightDB.commit()
        print(line)


def show():
    ''' It will show all the Contents in table flightData'''
    try:
        cursor.execute('SELECT * FROM flightData')
        data = cursor.fetchall()
        print(f"\n{line}\nDetails of Flight : ")
        for flights in data:
            print(' ', flights)
        print(line)
    except Exception:
        print("Empty Database! Please Insert Details of Flight using 1st Operation.")


def delete():
    '''To delete some details of particular flight'''
    flightNo = int(input(f"\n{line}\nPlease Enter Flight Number to delete its Data : "))
    try:
        cursor.execute(f'DELETE FROM flightData WHERE flightNo={flightNo}')
        print(f'{line}\nDetails of Flight-{flightNo}, deleted Successfully.')
        flightDB.commit()
        print(line)
    except Exception:
        print(f'Details of Flight-{flightNo} was Not Found!')


def search():
    '''It'll Search and return a particular Flight details '''
    flightNo = int(input(f"\n{line}\nPlease Enter Flight Number to Search : "))
    try:
        cursor.execute(f'SELECT * FROM flightData WHERE flightNo={flightNo}')
        data = cursor.fetchall()
        print(f'\n{line}\nDetails of Flight-{flightNo} :\n {data} \n{line}')
    except Exception:
        print(f'Details of Flight-{flightNo} was Not Found!')


def updateData():
    '''It'll Update the details of particular flight'''
    oldfNo = int(input(f"\n{line}\nEnter Flight Number to Update its detail : "))
    try:
        print(f'{line}\nEnter the New Details of Flight-{oldfNo} : ')
        flightNo = int(input(f'Enter flight Number :'))
        cursor.execute(f'UPDATE flightData SET flightNo={flightNo} WHERE flightNo={oldfNo}')
        fName = input(f'Enter Name of flight-{flightNo} : ')
        cursor.execute(f'UPDATE flightData SET fName="{fName}" WHERE flightNo={flightNo}')
        fare = input(f'Enter Fare of {fName}-{flightNo} : ')
        cursor.execute(f'UPDATE flightData SET fare={fare} WHERE flightNo={flightNo}')
        flightDB.commit()
        print(f'{line}\nDetails of {fName}-{flightNo} Updated Successfully.\n{line}')
    except Exception:
        print(f'Details of Flight-{oldfNo} was Not Found!')


while True:
    print('\nOperations :\n 1 - Add Flight Details.\n 2 - Show Flight Details\n 3 - Update Flight Details. ')
    print(' 4 - Delete Flight Detail.\n 5 - Search Flights.\n 6 - Clock.\n 7 - EXIT.')
    op = int(input("\nEnter Number to Choose Operation : "))
    if op == 1:
        write_data()
    elif op == 2:
        show()
    elif op == 3:
        updateData()
    elif op == 4:
        delete()
    elif op == 5:
        search()
    elif op == 6:
        import time
        live_time = time.strftime("%H:%M:%S")
        print(f"\n{line}\nThe Current timings are : {live_time} \n{line}\n")
    elif op == 7:
        print(f"\n{line}\nThanks for Using Airways Management System.\n")
        print(f"Creators of Airways Management System :\n | Akash kumar Singh |  Rohit kumar | Pawan Meena | \n{line}")
        break
    else:
        print("Please Choose CORRECT Operation!")

flightDB.close()




