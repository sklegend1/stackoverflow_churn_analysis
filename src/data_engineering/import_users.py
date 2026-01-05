import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from lxml import etree
import mysql.connector
from mysql.connector import errors  # برای گرفتن نوع خطا

# ---- تنظیمات اتصال به MySQL داکری ----
conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3307,
    user="root",
    password="rootpass",
    database="xml_db",
)

# cursor سریع معمولی، بدون prepared=True
cursor = conn.cursor()

BATCH_SIZE = 1000  # هر چندتا ردیف یکبار INSERT گروهی

def normalize_datetime(s):
    if not s:
        return None
    s = s.replace("T", " ")
    return s[:19]

XML_FILE = "xml/Users.xml"

context = etree.iterparse(XML_FILE, events=("end",), tag="row")

batch = []
total = 0

insert_sql = """
    INSERT IGNORE INTO users
      (Id, Reputation, CreationDate, DisplayName, LastAccessDate,
       WebsiteUrl, Location, AboutMe, Views, UpVotes, DownVotes, AccountId)
    VALUES
      (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

def insert_batch(batch):
    global total
    if not batch:
        return
    try:
        # تلاش سریع: کل batch با هم
        cursor.executemany(insert_sql, batch)
        conn.commit()
        total += len(batch)
        print(f"Inserted {total} rows...")
    except errors.ProgrammingError as e:
        print("Batch failed, falling back to row-by-row. Error:", e)
        # اگر batch مشکل داشت، سطر به سطر امتحان کن
        for row in batch:
            try:
                cursor.execute(insert_sql, row)
            except errors.ProgrammingError as e_row:
                # احتمال زیاد مشکل از AboutMe است → یک بار با NULL امتحان کن
                row_list = list(row)
                ABOUTME_INDEX = 7  # ترتیب: Id, Reputation, CreationDate, DisplayName, LastAccessDate, WebsiteUrl, Location, AboutMe, Views, UpVotes, DownVotes, AccountId
                original_about = row_list[ABOUTME_INDEX]
                row_list[ABOUTME_INDEX] = None
                try:
                    cursor.execute(insert_sql, tuple(row_list))
                    print(f"Row with Id={row[0]} inserted with AboutMe=NULL (had error before)")
                except errors.ProgrammingError as e2:
                    # اگر با NULL هم نشد، این سطر رو کلاً رد کن
                    print(f"Skipping row Id={row[0]} due to error:", e2)
                    # می‌تونی اینجا لاگ فایل هم بنویسی اگر دوست داری
                    continue
        conn.commit()
        total += len(batch)
        print(f"Inserted {total} rows (after per-row fallback).")


for _, elem in context:
    a = elem.attrib

    Id = int(a["Id"])
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
