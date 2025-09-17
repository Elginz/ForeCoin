# ForeCoin 

There are two bots currently. These are 
<a href="https://t.me/HostingSG_bot">@HostingSG_bot</a> and <a href="https://t.me/HostingSG_Owners_bot">@HostingSG_Owners_bot</a> 

They work differently. ```HostingSG_bot``` serves as a frontend for customers, while ```HostingSG_Owners_bot``` serves owners. 
They are both utilising the same mysql database. 
______
<h3 align="center">📚Dependencies/Libraries📖</h3>

Run the requirements.txt file 
- ```pip install -r requirements.txt```

**OR**

Run this in terminal to download the following libraries/dependencies  

- ```pip install python-telegram-bot aiohttp aiomysql mysql-connector-python pytz```


Make sure you're using python-telegram-bot v20+, because the code uses ```Application```, ```ContextTypes```, ```filters```, and ```ParseMode``` as per the v20+ API.
- ```pip install python-telegram-bot==20.7 mysql-connector-python```

*Some dependencies/libraries and their Reasons*
- ``` python-telegram-bot``` for telegram
- ``` aiomysql``` for async MySQL database connections
- ``` aiohttp``` to fetch the image over HTTP asynchronously
- ``` pytz``` For handling timezone (Asia/Singapore):

(May need to install this for the bot token. It is still hardcoded in the files, so storing in and ```.env``` file may be better for security)
- ```pip install python-dotenv```
______
<h3 align="center">🗃️Project Structure🗃️</h3>

This is the current <a href="https://www.canva.com/design/DAGcsg9Km-w/pKuvo6Kr6qjcPzAmaUUcPw/edit"> User and Host Flowchart</a> 

- Backups: *This is just a backup of the other files*
- random: *Files that i just keep as i may want to use it next time*
- owners: *Codes for Owners*

```
project/
│
├── Backups/            
│   ├── hostOwners copy.py
│   └── hostSG copy.py
│   ├── main_copy.sql
│   └── Menu Copy.py
├── random/            
│   ├── examples.sql
│   └── token.env
├── owners/             
│   ├── __init__.py
│   └── db.py            
│   └── Menu.py          
│   └── Profiles.py      
|
├── hostOwners.py  
├── hostSG.py   
├── main.sql   
└── requirements.txt
```
______

<h3 align="center">🖥️ How to run 🏃</h3>

First, create a mysql database within your pc, using the mysql script ```main.sql```

**HostingSG_boty**
- ```python hostSG.py```
  
**HostingSG_Owners_bot**
- ```python hostOwners.py```
______
<h3 align="center">🆘Current work in progress (help needed!)🆘</h3>

**Host Owners** 

Currently requires a timetable booking scheduler. 
- Hosts need to determine how their timeslots are seperated (```15 minutes```, ```30 minutes```, ```1 Hour```)
- Hosts can get a _prompt_ of user's bookings and decide if they can ```cancel```, ```reject``` ```accept``` them. They are able to accept more than 1 booking per time slot. (need to think about it again)

Here is a list of bot bookings for consideration

- <a href="https://www.youtube.com/watch?v=ykm2T2Jm8yc"> Youtube: Reservation Telegram Bot using Python</a> 
- <a href="https://github.com/AlexBaranovIT/Reservation-Telegram-bot/tree/main"> Github: Reservation-Telegram_bot</a> 
- <a href="https://github.com/AndrewAkhmetov/BookingTelegramBot"> Github: Booking Telegram Bot</a> 
______
**Host Users**
- Need to debug it as i changed host owners quite abit
- Users need to select the hosts based on the enquires provided to them
- users can select the respective time slots to book an appointment
- Users will get a _prompt_ if the host ```accept```/```reject```/```cancel``` their booking
______
**Payment Systems**

This can be done later or last. Here are some possible payment service providers for the platform.

This one seems quite interesting with a lot of integration possibilities to common payment methods in SG: https://hitpayapp.com/

Also they provide the full REST APIs  in their documentation: https://docs.hitpayapp.com/apis/overview , and they don' have a setup cost/subscription; only transaction cost. Might be interesting to check them out more.
