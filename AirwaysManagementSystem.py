# Airways Management System

'''
Make Sure you have installed two modules :
1. Python text-to-speech - pip install pyttsx3
2. Python My SQL Connector - pip install mysql-connector
'''

import mysql.connector    # it'll connect python to MySQL RDMS by which we can access our Databases.
import pyttsx3    # this is Python text-to-speech which gives a voice to our Airways Management System.


def speak(line):
    '''it will give a female voice to Every line given as argument(parameter)'''
    voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    voice = pyttsx3.init()
    voice.setProperty('voice', voice_id)
    voice.say(line)
    voice.runAndWait()


speak("Enter Sql USER Password to get Started & if you don't know the password just press Enter")
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
    speak('Enter the Number of Flights you want to insert in Database')
    No_Of_flight = int(input(f"\n{line}\nEnter the Number of Flights you want to insert : "))
    for num in range(No_Of_flight):
        speak(f'Enter the Details of {num+1}-Flight')
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
        speak('Showing the Details of Flights')
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
    speak('Please Enter Flight Number to delete its Data : ')
    flightNo = int(input(f"\n{line}\nPlease Enter Flight Number to delete its Data : "))
    try:
        cursor.execute(f'DELETE FROM flightData WHERE flightNo={flightNo}')
        speak(f'Details of Flight-{flightNo}, deleted Successfully')
        print(f'{line}\nDetails of Flight-{flightNo}, deleted Successfully.')
        flightDB.commit()
        print(line)
    except Exception:
        print(f'Details of Flight-{flightNo} was Not Found!')


def search():
    '''It'll Search and return a particular Flight details '''
    speak("Please Enter Flight Number to Search : ")
    flightNo = int(input(f"\n{line}\nPlease Enter Flight Number to Search : "))
    try:
        cursor.execute(f'SELECT * FROM flightData WHERE flightNo={flightNo}')
        data = cursor.fetchall()
        print(f'\n{line}\nDetails of Flight-{flightNo} :\n {data} \n{line}')
    except Exception:
        print(f'Details of Flight-{flightNo} was Not Found!')


def updateData():
    '''It'll Update the details of particular flight'''
    speak('Enter Flight Number to Update its detail :')
    oldfNo = int(input(f"\n{line}\nEnter Flight Number to Update its detail : "))
    try:
        speak(f'Enter the New Details of Flight-{oldfNo} :')
        print(f'{line}\nEnter the New Details of Flight-{oldfNo} : ')
        flightNo = int(input(f'Enter flight Number :'))
        cursor.execute(f'UPDATE flightData SET flightNo={flightNo} WHERE flightNo={oldfNo}')
        fName = input(f'Enter Name of flight-{flightNo} : ')
        cursor.execute(f'UPDATE flightData SET fName="{fName}" WHERE flightNo={flightNo}')
        fare = input(f'Enter Fare of {fName}-{flightNo} : ')
        cursor.execute(f'UPDATE flightData SET fare={fare} WHERE flightNo={flightNo}')
        flightDB.commit()
        speak(f'Details of {fName}-{flightNo} Updated Successfully.')
        print(f'{line}\nDetails of {fName}-{flightNo} Updated Successfully.\n{line}')
    except Exception:
        print(f'Details of Flight-{oldfNo} was Not Found!')


speak_one_time = 0
speak_one = 0
while True:
    if speak_one == 0:
        speak("Enter Number to choose specific operation you want to perform")
        speak_one = 1
    print('\nOperations :\n 1 - Add Flight Details.\n 2 - Show Flight Details\n 3 - Update Flight Details. ')
    print(' 4 - Delete Flight Detail.\n 5 - Search Flights.\n 6 - Clock.\n 7 - EXIT.')
    if speak_one_time == 0:
        speak("1 for Adding Flight Details, 2 for Displaying flight details, 3 to Update flight details"
              "4 for Deleting flight details, 5 to Search Flights, 6 to know current timings, and 7 for Exit")
        speak_one_time = 1

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
        speak(f"The Current timings are : {live_time}")
        print(f"\n{line}\nThe Current timings are : {live_time} \n{line}\n")
    elif op == 7:
        speak('Thanks for Using Airways Management System')
        print(f"\n{line}\nThanks for Using Airways Management System.\n")
        speak("Creators of this, Airways Management System are Akash kumar Singh, Rohit Kumar and Pawan Meena ")
        print(f"Creators of Airways Management System :\n | Akash kumar Singh |  Rohit kumar | Pawan Meena | \n{line}")
        break
    else:
        speak("Please Choose CORRECT Operation!")
        print("Please Choose CORRECT Operation!")

flightDB.close()
