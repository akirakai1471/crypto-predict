"""Crypto headlines from public RSS: collected, timestamped, shown - not predicted from.

Free feeds keep a few days of items, so there is no history to put a news
feature through the walk-forward gates every price feature has passed. This
package therefore does three things and deliberately not a fourth: it collects
headlines with a point-in-time timestamp of our own, it shows them (dashboard,
CLI, optional alerts), and it computes leak-proof per-bar counts so their value
can be measured once months have accumulated. It does not feed the model.
"""
