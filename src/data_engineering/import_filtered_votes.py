import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from lxml import etree
import mysql.connector

# ---------- 1) اتصال به MySQL لوکال ----------
conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3307,          # اگر پورت داکرت چیز دیگه‌ایه اینو عوض کن
    user="root",
    password="rootpass",
    database="xml_db",  # دیتابیس لوکال که posts و filtered_votes توش هستن
)

cursor = conn.cursor()

# ---------- 2) گرفتن همه idهای جدول posts ----------

print("Loading post IDs from local posts table...")

cursor.execute("SELECT id FROM posts")
post_ids = set()
for (pid,) in cursor:
    post_ids.add(pid)

print(f"Loaded {len(post_ids)} post IDs.")

# ---------- 3) آماده‌سازی INSERT برای filtered_votes ----------

BATCH_SIZE = 1000
XML_FILE = "xml/Votes.xml"   # مسیر Votes.xml (یا هر اسمی داری)

def normalize_datetime(s):
    if not s:
        return None
    s = s.replace("T", " ")
    return s[:19]

insert_sql = """
    INSERT IGNORE INTO filtered_votes
      (Id, PostId, VoteTypeId, UserId, CreationDate, BountyAmount)
    VALUES
      (%s, %s, %s, %s, %s, %s)
"""

batch = []
total_inserted = 0

# ---------- 4) استریم XML و فیلتر بر اساس PostId ----------

print("Parsing XML and inserting filtered votes...")

context = etree.iterparse(XML_FILE, events=("end",), tag="row")

for _, elem in context:
    a = elem.attrib

    try:
        vote_id = int(a["Id"])
        post_id = int(a["PostId"])
    except (KeyError, ValueError):
        # اگر Id یا PostId مشکل داشت، این سطر رو رد کن
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        continue

    # فقط برای پست‌هایی که قبلاً فیلتر شده‌اند
    if post_id not in post_ids:
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        continue

    vote_type_id = int(a.get("VoteTypeId", 0)) if a.get("VoteTypeId") else 0

    user_id = None
    if "UserId" in a and a["UserId"] != "":
        try:
            user_id = int(a["UserId"])
        except ValueError:
            user_id = None

    creation_date = normalize_datetime(a.get("CreationDate"))

    bounty_amount = None
    if "BountyAmount" in a and a["BountyAmount"] != "":
        try:
            bounty_amount = int(a["BountyAmount"])
        except ValueError:
            bounty_amount = None

    batch.append(
        (
            vote_id,
            post_id,
            vote_type_id,
            user_id,
            creation_date,
            bounty_amount,
        )
    )

    if len(batch) >= BATCH_SIZE:
        cursor.executemany(insert_sql, batch)
        conn.commit()
        total_inserted += len(batch)
        print(f"Inserted {total_inserted} rows into filtered_votes...")
        batch.clear()

    # خالی کردن نود از حافظه
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

# ---------- 5) بچ آخر ----------

if batch:
    cursor.executemany(insert_sql, batch)
    conn.commit()
    total_inserted += len(batch)
    print(f"Inserted {total_inserted} rows into filtered_votes (final batch).")
    batch.clear()

cursor.close()
conn.close()

print("Done.")
print(f"Total filtered_votes inserted: {total_inserted}")
