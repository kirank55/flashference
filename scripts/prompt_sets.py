#!/usr/bin/env python3
"""Fixed prompt sets for single-reply experiments (S2/S3). Writes prompts/prompts.json when run."""
import json, os

PASSAGE = ("The Harbour Lights festival returns to the city of Marlow on the second weekend of June. "
 "Organisers expect around 42,000 visitors over three days, up from 35,500 last year. "
 "More than sixty lanterns, some taller than a bus, will be floated along the river each evening at 9 p.m. "
 "Entry is free, but tickets are required for the night cruises, which cost 18 pounds for adults and 9 pounds for children. "
 "The council has added two extra shuttle buses from the railway station, running every ten minutes until midnight. "
 "Local baker Maria Okafor says her stall will sell 3,000 lantern-shaped biscuits, and she has hired four temporary helpers.")

TYPOS = ("Yesterday the commitee met to discus the new reciept policy for the volunteer kitchen. "
 "Evryone agreed that every donation shoud be recorded in the ledgar, and that a second person must check the totals befor the end of the day. "
 "Treasurer Jonas Albrecht noted that last mounth three entries were mising, wich made the quarterly report late. "
 "The group also decided to purchace a small lockbox, and to review the arangement again in september.")

JSON_IN = '''{
  "id": 4821,
  "name": "Ravi Menon",
  "email": "ravi.menon@example.com",
  "role": "analyst",
  "team": "forecasting",
  "location": {"city": "Pune", "country": "India", "timezone": "Asia/Kolkata"},
  "skills": ["python", "sql", "statistics", "dashboards"],
  "active": true,
  "start_date": "2021-03-15"
}'''

CODE = '''import csv

def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def average_price(rows):
    total = 0
    for row in rows:
        total += float(row["price"])
    return total / len(rows) + 1

def most_expensive(rows):
    best = rows[0]
    for row in rows:
        if float(row["price"]) < float(best["price"]):
            best = row
    return best

if __name__ == "__main__":
    data = load_rows("items.csv")
    print("Average price:", average_price(data))
    print("Most expensive:", most_expensive(data)["name"])'''

EMAIL = ("Subject: Team offsite planning\n\nHi all,\n\nThe offsite is confirmed for Tuesday the 14th at the Lakeside Conference Centre. "
 "Please arrive by 9:00 so we can start the first session on time. Lunch will be provided, and vegetarian and vegan options are available. "
 "If you have any dietary restrictions, reply to this email by Friday. Parking is free on site, and a shuttle leaves the main office at 8:15.\n\n"
 "Best regards,\nPriya")

LIST = ("- Buy oat milk and two litres of sparkling water\n- Return the library books before the 20th\n- Call the plumber about the kitchen tap\n"
 "- Renew the car insurance policy\n- Print the boarding passes for the Lisbon trip\n- Water the balcony plants twice a week\n- Book a dentist appointment for November")

PROMPTS = {
 "copy": [
  {"id": "C1", "n_predict": 256, "user": "Fix all spelling mistakes in the following paragraph and output only the corrected paragraph, changing nothing else:\n\n" + TYPOS},
  {"id": "C2", "n_predict": 256, "user": "Here is a JSON record. Change the \"role\" field to \"senior analyst\" and the \"team\" field to \"risk\". Output the complete updated JSON and nothing else.\n\n" + JSON_IN},
  {"id": "C3", "n_predict": 320, "user": "The following Python program has two bugs (the average is off by one and the most expensive item is wrong). Fix the bugs and print the complete corrected file in a code block.\n\n" + CODE},
  {"id": "C4", "n_predict": 256, "user": "Quote word for word the sentences from the passage below that contain a number, one per line, with no other text.\n\n" + PASSAGE},
  {"id": "C5", "n_predict": 256, "user": "Rewrite the email below, replacing \"Tuesday the 14th\" with \"Thursday the 16th\" and the shuttle time with 8:30. Keep every other word identical and output only the new email.\n\n" + EMAIL},
  {"id": "C6", "n_predict": 256, "user": "Convert this bulleted list into a numbered list, keeping each item's text exactly the same. Output only the list.\n\n" + LIST},
 ],
 "free": [
  {"id": "F1", "n_predict": 256, "user": "Write a ~400-word explanation of how electric cars work."},
  {"id": "F2", "n_predict": 256, "user": "Write a short story about a lighthouse keeper who finds a message in a bottle."},
  {"id": "F3", "n_predict": 256, "user": "Write a short essay on the advantages and disadvantages of remote work."},
  {"id": "F4", "n_predict": 256, "user": "Explain how vaccines train the immune system, in plain language."},
  {"id": "F5", "n_predict": 256, "user": "Explain the difference between TCP and UDP and when you would use each."},
  {"id": "F6", "n_predict": 256, "user": "Describe what a day in the life of a medieval blacksmith might look like."},
 ],
}

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "prompts"), exist_ok=True)
    json.dump(PROMPTS, open(os.path.join(root, "prompts", "prompts.json"), "w"), indent=1)
    print("written")
