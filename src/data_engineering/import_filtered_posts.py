import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import csv
import mysql.connector

# ---------- 1) خواندن UserId ها از CSV ----------

CSV_FILE = r"processed-data/nmf_clusters_continent_pronouns_age_20.csv"   # مسیر CSV خودت
CSV_ID_COLUMN = "userid"              # اسم ستون یوزر در CSV

target_user_ids = set()

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        val = row.get(CSV_ID_COLUMN)
        if not val:
            continue
        try:
            uid = int(val)
        except ValueError:
            continue
        target_user_ids.add(uid)

print(f"Loaded {len(target_user_ids)} user IDs from CSV.")


# ---------- 2) اتصال به دیتابیس اصلی (سرور) و دیتابیس لوکال ----------

# کانکشن به سرور اصلی (جایی که جدول بزرگ posts هست):
remote_conn = mysql.connector.connect(
    host="external.abriment.com",        # ← آدرس سرور اصلی
    port=32038,                  # اگر پورت خاصی دارن، همین‌جا عوض کن
    user="Sohrab",           # یوزر دیتابیس اصلی
    password="Sohrab!@#KH123",   # پسورد دیتابیس اصلی
    database="data",           # اسم دیتابیس اصلی
)

# کانکشن به دیتابیس لوکال (داکر)
local_conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3307,                  # اگر مثل قبل map کردی
    user="root",
    password="rootpass",
    database="xml_db",          # همونی که جدول posts جدید رو توش ساختی
)

# remote_cursor رو بدون buffer می‌گیریم که رم نخوره
remote_cursor = remote_conn.cursor(dictionary=True)
local_cursor = local_conn.cursor()

# ---------- 3) آماده‌سازی INSERT لوکال ----------

insert_sql = """
    INSERT IGNORE INTO posts
      (id, posttypeid, acceptedanswerid, creationdate,
       score, parentid, viewcount, owneruserid,
       lasteditoruserid, lasteditordisplayname,
       lasteditdate, lastactivitydate, tags,
       answercount, commentcount, favoritecount,
       communityowneddate)
    VALUES
      (%s, %s, %s, %s,
       %s, %s, %s, %s,
       %s, %s,
       %s, %s, %s,
       %s, %s, %s,
       %s)
"""

REMOTE_BATCH_SIZE = 50000   # چند تا ردیف از سرور اصلی در هر مرحله
LOCAL_BATCH_SIZE = 2000     # چند تا ردیف هر بار تو لوکال اینسرت کنیم

last_id = 4953705
total_inserted = 1443870

while True:
    # 4) خواندن یک بچ از جدول اصلی
    remote_cursor.execute(
        """
        SELECT
          id,
          posttypeid,
          acceptedanswerid,
          creationdate,
          score,
          parentid,
          viewcount,
          owneruserid,
          lasteditoruserid,
          lasteditordisplayname,
          lasteditdate,
          lastactivitydate,
          tags,
          answercount,
          commentcount,
          favoritecount,
          communityowneddate
        FROM posts
        WHERE id > %s
        ORDER BY id ASC
        LIMIT %s
        """,
        (last_id, REMOTE_BATCH_SIZE)
    )
    rows = remote_cursor.fetchall()

    if not rows:
        print("No more rows from remote. Done.")
        break

    batch_for_local = []

    for r in rows:
        owner = r["owneruserid"]
        # بعضی پست‌ها owner ندارن (مثلاً ویکی/Community)
        if owner is None:
            continue
        if owner not in target_user_ids:
            continue

        batch_for_local.append((
            r["id"],
            r["posttypeid"],
            r["acceptedanswerid"],
            r["creationdate"],
            r["score"],
            r["parentid"],
            r["viewcount"],
            r["owneruserid"],
            r["lasteditoruserid"],
            r["lasteditordisplayname"],
            r["lasteditdate"],
            r["lastactivitydate"],
            r["tags"],
            r["answercount"],
            r["commentcount"],
            r["favoritecount"],
            r["communityowneddate"],
        ))

        # اگر batch_for_local خیلی بزرگ شد، تخلیه‌اش کن
        if len(batch_for_local) >= LOCAL_BATCH_SIZE:
            local_cursor.executemany(insert_sql, batch_for_local)
            local_conn.commit()
            total_inserted += len(batch_for_local)
            print(f"Inserted {total_inserted} posts so far...")
            batch_for_local.clear()

    # بچ آخر همین حلقه
    if batch_for_local:
        local_cursor.executemany(insert_sql, batch_for_local)
        local_conn.commit()
        total_inserted += len(batch_for_local)
        print(f"Inserted {total_inserted} posts so far... (end of chunk)")
        batch_for_local.clear()

    # آپدیت کردن last_id برای مرحله بعد
    last_id = rows[-1]["id"]
    print(f"Continue from id > {last_id}")

remote_cursor.close()
local_cursor.close()
remote_conn.close()
local_conn.close()

print(f"Done. Total inserted posts: {total_inserted}")
