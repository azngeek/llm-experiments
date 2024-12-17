import csv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import (
    ChatOpenAI
)

from tabulate import tabulate


OPENAI_API_KEY=""
OPENAI_MODEL="gpt-4-turbo"

def llm(model: str = "gpt-4-turbo"):
    """
    Helper function which wraps OpenAI. At the current state we will only use OpenAI as this is the only capable model
    which can execute tool functions. All other models are not production ready yet.

    :param temperature:
    :param model:
    :return:
    """
    return ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            PERSON: A skilled music data analyst experienced in working with CSV files and music-related datasets.

            CONTEXT: You have a CSV file containing music data with the columns: id, artist, genre, and label.
            The release dates for the songs are not currently included and need to be included. If you do not know the exact
            relase_date, use your best guess.
            Your goal is to add release dates for each song as a new column 'release_date'.

            FORMAT: You will return a new List as CSV-String with the columns 'id' and the newly created column 'release_date'
            
            LIST: {list}
            """,
        )
    ]
)

music_data = """
id,artist,genre,label
1,Mariah Carey,Pop,Sony Music
2,Wham!,Pop,Sony Music
3,Shakin' Stevens,Rockabilly,Sony Music
4,Brenda Lee,Rock 'n' Roll,Universal
5,Kelly Clarkson,Pop,RCA
6,Rosé / Bruno Mars,Pop,R&B,Atlantic
7,José Feliciano,Latin Pop,RCA
8,Dean Martin,Jazz / Swing,Capitol
9,Melanie Thornton,Pop,X-Cell Records
10,Ariana Grande,Pop,Universal
11,Sia,Pop,Atlantic
12,Michael Bublé,Jazz / Pop Standards,Reprise Records
13,Chris Rea,Rock,Warner Music International
14,Sosa La M / Luciano,Hip-Hop,RCA & GOLD LEAGUE
15,Ed Sheeran & Elton John,Pop,Atlantic
16,Linkin Park,Alternative Rock,Warner Music International
17,John & Yoko / The Plastic Ono Band with The Harlem Community Choir,Rock,EMI
18,Nina Chuba,Pop,Jive
19,Bobby Helms,Country,Geffen Records
20,Paul McCartney,Pop,Capitol

"""

music_sorter = prompt | llm(model="gpt-4o")

response = music_sorter.invoke({"list": music_data})
print(response.content)

# convert list with release date in a dictionary Dictionary {id: release_date}
release_dates = {}
for line in response.content.strip().split("\n")[1:]:  # [1:] Skip Header
    id_, date = line.split(",")
    release_dates[int(id_)] = date

print(release_dates)

# convert music_list to list and add column 'release_date'
reader = csv.DictReader(music_data.strip().split("\n"))
chart_data = []

for row in reader:
    row_id = int(row["id"])
    row["release_date"] = release_dates.get(row_id, None)
    chart_data.append(row)

# Sort by release_date
sorted_chart_data = sorted(chart_data, key=lambda x: x["release_date"])

# create table for output
table = [[row["id"], row["artist"], row["genre"], row["label"], row["release_date"]] for row in sorted_chart_data]

# print tabulate table
headers = ["ID", "Artist", "Genre", "Label", "Release Date"]
print(tabulate(table, headers=headers, tablefmt="grid"))
