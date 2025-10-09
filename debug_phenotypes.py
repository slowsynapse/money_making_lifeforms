"""Debug script to check phenotype data in database."""
import sqlite3

db_path = "/home/agent/workdir/evolution/cells.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check cell 19 phenotypes
print("=== Cell #19 Phenotypes ===")
cursor.execute("""
    SELECT cell_id, timeframe, total_profit, total_trades, win_rate
    FROM cell_phenotypes
    WHERE cell_id = 19
""")
rows = cursor.fetchall()
for row in rows:
    print(f"  Cell {row[0]}, {row[1]}: ${row[2]:.2f}, {row[3]} trades, {row[4]*100 if row[4] else 0:.0f}% win rate")

if not rows:
    print("  ⚠️  No phenotypes found for Cell #19")

# Check if any phenotypes have null total_trades
print("\n=== Checking for NULL total_trades ===")
cursor.execute("""
    SELECT cell_id, timeframe, total_trades
    FROM cell_phenotypes
    WHERE total_trades IS NULL
    LIMIT 10
""")
null_rows = cursor.fetchall()
if null_rows:
    print(f"  Found {len(null_rows)} phenotypes with NULL total_trades:")
    for row in null_rows:
        print(f"    Cell {row[0]}, {row[1]}: total_trades = {row[2]}")
else:
    print("  ✓ No NULL total_trades found")

# Check if any phenotypes have null win_rate
print("\n=== Checking for NULL win_rate ===")
cursor.execute("""
    SELECT cell_id, timeframe, total_trades, win_rate
    FROM cell_phenotypes
    WHERE win_rate IS NULL
    LIMIT 10
""")
null_win_rate = cursor.fetchall()
if null_win_rate:
    print(f"  ⚠️  Found {len(null_win_rate)} phenotypes with NULL win_rate:")
    for row in null_win_rate:
        print(f"    Cell {row[0]}, {row[1]}: total_trades={row[2]}, win_rate={row[3]}")
else:
    print("  ✓ No NULL win_rate found")

# Check total phenotypes
cursor.execute("SELECT COUNT(*) FROM cell_phenotypes")
count = cursor.fetchone()[0]
print(f"\n=== Total phenotypes in database: {count} ===")

# Sample a few cells
print("\n=== Sample cells and their phenotypes ===")
cursor.execute("""
    SELECT c.cell_id, c.fitness, c.dsl_genome, COUNT(p.phenotype_id)
    FROM cells c
    LEFT JOIN cell_phenotypes p ON c.cell_id = p.cell_id
    GROUP BY c.cell_id
    ORDER BY c.cell_id
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"  Cell #{row[0]}: fitness=${row[1]:.2f}, phenotypes={row[3]}, strategy={row[2][:50]}...")

conn.close()
