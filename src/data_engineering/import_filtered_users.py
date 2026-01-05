import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from lxml import etree
import mysql.connector
from mysql.connector import errors
import csv

# ---------- 1) خواندن UserId و cluster_nmf_20 از CSV ----------

CSV_FILE = r"processed-data/nmf_clusters_continent_pronouns_age_20.csv"      # مسیر CSV خودت
CSV_ID_COLUMN = "userid"                # اسم ستون ID در CSV
CSV_CLUSTER_COLUMN = "cluster_nmf_20"   # اسم ستون کلاستر در CSV

cluster_map = {}   # { user_id(int): cluster_nmf_20(int یا None) }

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        id_val = row.get(CSV_ID_COLUMN)
        cl_val = row.get(CSV_CLUSTER_COLUMN)
        if not id_val:
            continue
        try:
            uid = int(id_val)
        except ValueError:
            continue

        if cl_val is None or cl_val == "":
            cluster_map[uid] = None
        else:
            try:
                cluster_map[uid] = int(cl_val)   # اگر float داری، اینجا float(...)
            except ValueError:
                cluster_map[uid] = None

print(f"Loaded {len(cluster_map)} users with cluster_nmf_20 from CSV")


# ---------- 2) اتصال به MySQL ----------

conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3307,
    user="root",
    password="rootpass",
    database="xml_db",
)

cursor = conn.cursor()
BATCH_SIZE = 1000
total = 0  # برای شمارش کل ردیف‌های اینسرت شده

def normalize_datetime(s):
    if not s:
        return None
    s = s.replace("T", " ")
    return s[:19]

XML_FILE = "xml/Users.xml"

context = etree.iterparse(XML_FILE, events=("end",), tag="row")

insert_sql = """
    INSERT IGNORE INTO filtered_users
      (Id, Reputation, CreationDate, DisplayName, LastAccessDate,
       WebsiteUrl, Location, AboutMe, Views, UpVotes, DownVotes, AccountId,
       ClusterNmf20)
    VALUES
      (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

batch = []

def insert_batch(batch):
    global total
    if not batch:
        return
    try:
        cursor.executemany(insert_sql, batch)
        conn.commit()
        total += len(batch)
        print(f"Inserted {total} rows...")
    except errors.ProgrammingError as e:
        print("Batch failed, falling back to row-by-row. Error:", e)
        ABOUTME_INDEX = 7  # جای AboutMe توی tuple
        for row in batch:
            try:
                cursor.execute(insert_sql, row)
            except errors.ProgrammingError as e_row:
                row_list = list(row)
                original_about = row_list[ABOUTME_INDEX]
                row_list[ABOUTME_INDEX] = None
                try:
                    cursor.execute(insert_sql, tuple(row_list))
                    print(f"Row with Id={row[0]} inserted with AboutMe=NULL")
                except errors.ProgrammingError as e2:
                    print(f"Skipping row Id={row[0]} due to error:", e2)
                    continue
        conn.commit()
        total += len(batch)
        print(f"Inserted {total} rows (after per-row fallback).")


# ---------- 3) استریم XML + فیلتر بر اساس CSV + اضافه کردن کلاستر ----------

for _, elem in context:
    a = elem.attrib

    Id = int(a["Id"])

    # فقط یوزرهایی که در CSV داریم:
    if Id not in cluster_map:
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        continue

    cluster_val = cluster_map[Id]

    Reputation = int(a.get("Reputation", 0))
    CreationDate = normalize_datetime(a.get("CreationDate"))
    DisplayName = a.get("DisplayName")
    LastAccessDate = normalize_datetime(a.get("LastAccessDate"))
    WebsiteUrl = a.get("WebsiteUrl")
    Location = a.get("Location")
    AboutMe = a.get("AboutMe")
    Views = int(a.get("Views", 0))
    UpVotes = int(a.get("UpVotes", 0))
    DownVotes = int(a.get("DownVotes", 0))
    AccountId = int(a["AccountId"]) if "AccountId" in a else None

    batch.append((
        Id,
        Reputation,
        CreationDate,
        DisplayName,
        LastAccessDate,
        WebsiteUrl,
        Location,
        AboutMe,
        Views,
        UpVotes,
        DownVotes,
        AccountId,
        cluster_val,   # ستون جدید در انتهای tuple
    ))

    if len(batch) >= BATCH_SIZE:
        insert_batch(batch)
        batch.clear()

    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

# بچ آخر
if batch:
    insert_batch(batch)
    batch.clear()

cursor.close()
conn.close()
print("Done.")
