"""Bounded source-role checks, not a general natural-language truth verifier.

Only complete, recognized balance/payment sentences may be excluded. Mixed or
unrecognized refund statements must supply a literal refund binding or stop
scoring. No tool truth is used to repair the assistant's asserted amount.
"""

import re
from decimal import Decimal


AMOUNT = r"\$?(?P<amount>\d+(?:,\d{3})*\.\d{1,2})(?!\d)"
CARD = r"(?:your|the) gift card(?: \(gift card ending in \d+\))?"
MONEY = r"\$\d+(?:,\d{3})*\.\d{1,2}(?!\d)"


def normalized(text):
    return " ".join(text.replace("*", "").split()).strip()


def non_refund_scope(text):
    """Return an audited role only for a WHOLE sentence in a closed grammar."""
    text = normalized(text)
    patterns = {
        "CARD_BALANCE_AND_PAYMENT_INSTRUMENT": (
            rf"Since {CARD} has a balance of {MONEY}, "
            r"it's available to receive the refund\."
        ),
        "REFERENCED_REFUND_WITH_CURRENT_BALANCE": (
            rf"This amount will be refunded to {CARD}, "
            rf"which currently has a balance of {MONEY}\."
        ),
        "POST_REFUND_CARD_BALANCE": (
            rf"After the refund, {CARD} will have a balance of {MONEY}\."
        ),
        "PAYMENT_DUE_NOT_REFUND": (
            rf"(?:Since {MONEY} > {MONEY}, )?"
            rf"you (?:would actually need to|need to|must) pay an additional {MONEY} "
            r"for the new (?:lamp|item), not receive a refund\."
        ),
    }
    for role, pattern in patterns.items():
        if re.fullmatch(pattern, text, re.I):
            return role
    return None


def literal_refund_amounts(text):
    """Extract supported positive refund mentions, preserving incorrect values.

    An unfamiliar negation/condition is deliberately not guessed. A failed binding
    is an extractor error, never a factual MISMATCH or automatic success.
    """
    text = normalized(text)
    if re.search(
        r"\b(?:not|never|no|won't|cannot)\s+(?:(?:receive|get|issue|be|a|any)\s+)*refund"
        r"|\b(?:if|unless|might|may)\b",
        text,
        re.I,
    ):
        return set()
    patterns = [
        # Refunded $X; refund amount will be $X; refund of the price difference ($X).
        rf"\brefund(?:ed|ing)?\s*"
        rf"(?:(?:amount|for this modification|will be|would be|is|of|the price difference)\s*)*"
        rf"[:(]?\s*{AMOUNT}",
        rf"{AMOUNT}\s+(?:(?:to be|will be|would be)\s+)?refund(?:ed)?\b",
        # The number explicitly named as the difference, not either input price.
        rf"\bprice difference\s+(?:would be|will be|is)\s+{AMOUNT}"
        r"\s*\([^()]*\), which (?:would|will) be refunded\b",
        # Bind the named refund amount, never the two comparison prices that follow.
        rf"\brefund\s*:\s*the\s+{AMOUNT}\s+amount\s+is\s+the\s+"
        r"full\s+price\s+difference\b",
        # A qualified amount assertion remains an assertion to verify against tools.
        rf"\brefund\s+appears\s+to\s+be\s+{AMOUNT}",
        # Explicit price-difference subject followed by a refund predicate.
        # The optional replacement price is context, never the bound amount.
        rf"\bprice difference\s+(?:is|of)\s+{AMOUNT}"
        rf"(?: to the replacement (?:desk lamp|item) \(item \d+ at {MONEY}\))?"
        r"(?: \(since the new item is cheaper\))?"
        r"(?:, which)? (?:is|will be|would be) refunded\b",
        rf"\bprice difference\s*:\s*{AMOUNT}\s*\(refund(?:ed)? to\b",
        # Explicitly asserted repayment of the original payment must still be
        # verified: in these tasks it can be a real, incorrect refund claim.
        rf"\boriginal payment of\s+{AMOUNT}\s+to (?:your |the )?gift card"
        r"(?: gift card ending in \d+)? (?:will be|would be|is) refunded\b",
        r"\brefund amount (?:would be|will be|is) the original payment "
        rf"for the item being replaced\s*\({AMOUNT} to (?:the|your) gift card\)",
        rf"\bis being refunded the amount of the modification\s*\({AMOUNT}\)",
        rf"\btotal refund to (?:your |the )?gift card\s*:\s*{AMOUNT}",
        # For a stated total A (role) + B (role) = C, bind C literally.
        # Do not recompute C or confuse either operand with the total claim.
        rf"\btotal refund to (?:your |the )?gift card (?:would be|will be|is) "
        rf"{MONEY} \(original (?:item|payment) refund\)\s*\+\s*"
        rf"{MONEY} \(price difference refund\)\s*=\s*{AMOUNT}",
    ]
    values = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            values.add(Decimal(match["amount"].replace(",", "")))
    return values


def check_refund_binding(value, text):
    if not isinstance(value, str):
        raise ValueError("Refund value must be a decimal string")
    amount = Decimal(value)
    if not amount.is_finite() or amount not in literal_refund_amounts(text):
        raise ValueError("Refund amount lacks an unambiguous source-role binding")


def check_refund_coverage(claims, text):
    asserted = literal_refund_amounts(text)
    extracted = {Decimal(c["value"]) for c in claims if c["kind"] == "refund_amount"}
    if not asserted.issubset(extracted):
        raise ValueError("Explicit refund amount omitted from candidate")
