"""Read-only SQLite tool bound to a sandbox database.

================================================================================
【这个文件演示的是"黑名单的局限"——一个很好的面试素材】
================================================================================

三层防御，从弱到强：

    第 1 层  只允许 SELECT 开头     ← 可绕过
    第 2 层  禁关键词黑名单         ← 可绕过
    第 3 层  只读连接 mode=ro      ← ★ 真正可靠的那一层

    前两层是"看起来像防御"，第三层是"架构上不可写"。
    面试时可以讲：应用层过滤永远只是第一道门槛，
    真正的安全来自"让危险操作在物理上不可能发生"。

【为什么保留可绕过的第 2 层】
    不是为了安全，是为了**给 R-04 规则提供检测素材**。
    如果第一层挡得死死的，"UNION SELECT" 这个攻击根本到不了数据库，
    也就没有轨迹可审计了。靶场需要保留攻击面。
"""

from __future__ import annotations

import os
import sqlite3

# 单次查询返回的最大行数。防止 "select * from 大表" 把上下文撑爆。
MAX_ROWS = 20

# Statements that would let a model escalate beyond a read. Checked before the
# connection is opened so a denial never touches the file.
#
# 【这个黑名单挡不住什么】
#   · union select          —— 不在列表里，能穿透（R-04 专门抓它）
#   · 用注释变形            —— "sel/**/ect" 会被 startswith 拦下，但
#                              "select 1 -- ; drop table x" 呢？注释后的部分
#                              不执行，所以其实安全；反过来
#                              "select 1; drop table x" 会被堆叠查询正则抓到
#   · 大小写变形            —— 已用 .lower() 处理
#   · 全角字符              —— 未处理
#
# ★ 记住这个结论：黑名单永远有漏。所以第 3 层（只读连接）才是底线。
FORBIDDEN_PREFIXES = ("insert", "update", "delete", "drop", "alter", "attach", "pragma")


def sql_query(workspace: str, sql: str) -> str:
    statement = sql.strip()
    lowered = statement.lower()

    # ---- 第 1 层：必须以 select 开头 ----
    # 为什么先 strip？模型可能输出 "\n  SELECT ..."，不 strip 会误判。
    if not lowered.startswith("select"):
        return "[sql_query] rejected: only SELECT statements are allowed"

    # ---- 第 2 层：关键词黑名单 ----
    # 注意这是子串匹配（in），不是前缀匹配。所以 "select 1; drop table x"
    # 里的 "drop" 会被抓到 —— 这正是堆叠查询的防御点。
    # 副作用：如果表名里有 "update_log"，对它的正常查询也会被拒。
    # 这是"宽召回"的典型代价，权衡后接受（因为拒绝对审计无害，漏掉才有害）。
    if any(keyword in lowered for keyword in FORBIDDEN_PREFIXES):
        # Chained statements such as "select 1; drop table x" land here.
        return "[sql_query] rejected: statement contains a forbidden keyword"

    db_path = os.path.join(workspace, "demo.db")
    if not os.path.isfile(db_path):
        return f"[sql_query] no such database: demo.db"

    try:
        # ---- 第 3 层：只读连接（真正的防线） ----
        #
        # mode=ro 让 SQLite 在**文件系统层面**以只读方式打开数据库：
        # 即使前面两层被绕过，写操作也会在 SQLite 内核里报
        # "attempt to write a readonly database"。
        #
        # 【uri=True 是必须的】
        #   不加这个参数，sqlite3 会把 "file:xxx?mode=ro" 当成一个
        #   字面文件路径，然后报 "unable to open database file"。
        #   这个坑很常见，调试时会浪费半小时。
        #
        # 【timeout=5】
        #   防止另一个进程持有写锁时这里永久阻塞。
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as exc:
        return f"[sql_query] connection error: {exc}"

    try:
        cursor = connection.execute(statement)
        # 不用 fetchall：MAX_ROWS 限制了读取量，
        # 大表查询不会一次性把几万行拉进内存
        rows = cursor.fetchmany(MAX_ROWS)
        # description 是 [(name, type, ...), ...]，只取第 0 位拼列名
        columns = [description[0] for description in cursor.description or []]
    except sqlite3.Error as exc:
        return f"[sql_query] error: {exc}"
    finally:
        # 必须关连接。放在 finally 保证异常时也执行。
        connection.close()

    if not rows:
        # 空结果也要返回列名，让模型知道查询执行成功了
        return f"[sql_query] 0 rows; columns={columns}"
    # 每行渲染成 dict 形式，可读性比 tuple 好，
    # 而且模型更容易理解"哪一列是什么"
    rendered = "\n".join(str(dict(zip(columns, row))) for row in rows)
    return f"[sql_query] {len(rows)} rows; columns={columns}\n{rendered}"

