from kestrel import text as T
from kestrel.data import fix_mojibake


def test_mojibake_is_repaired():
    assert fix_mojibake("urgÃ©nt: installer did not turn up") == "urgent: installer did not turn up"
    assert fix_mojibake("purifier not working Ã¢â‚¬Â¦") == "purifier not working ..."
    assert fix_mojibake("pls hÃ©lp Ã¢â‚¬â€œ issue") == "pls help - issue"
    assert fix_mojibake("plain text") == "plain text"
    assert fix_mojibake(None) == ""


def test_greeting_ids_closings_and_payment_are_stripped():
    s = "namaste, motor of air fryer burnt smell paid on upi order KO7654321 pls call back"
    assert T.clauses(s) == ["motor of P burnt smell"]
    assert T.payment_remark(s) == "paid on upi"


def test_two_requests_are_split_and_last_one_is_kept_last():
    s = "pls help - is ceiling fan safe for kids, ceiling fan leaking water from bottom paid by emi kindly resolve"
    assert T.clauses(s) == ["is P safe for kids", "P leaking water from bottom"]
    assert T.last_clause(s) == "P leaking water from bottom"


def test_two_part_single_requests_are_not_split():
    assert T.clauses("hi, air fryer arrived damaged, need replacement thanks") == ["P arrived damaged, need replacement"]
    assert T.clauses("warranty rejected for mixer, why") == ["warranty rejected for P, why"]
    assert T.clauses("sir box was open, fan scratched") == ["box was open, P scratched"]


def test_display_clauses_keep_product_words():
    s = "good morning, want AMC filter kit for purifier, purifier arrived damaged, need replacement"
    assert T.clauses(s, mask_products=False) == ["want amc filter kit for purifier",
                                                 "purifier arrived damaged, need replacement"]


def test_please_call_back_regarding_is_vague_not_a_closing():
    # Regression: "please call back" used to be stripped as a sign-off, leaving "regarding P".
    assert T.clauses("sir please call back regarding product reg no SR12345") == ["please call back regarding P"]
    assert T.is_vague("sir please call back regarding product reg no SR12345")


def test_vague_is_decided_by_the_last_request():
    assert T.is_vague("box was open, purifier scratched, please call back regarding purifier")
    assert not T.is_vague("please call back regarding purifier, purifier leaking water from bottom")
    assert not T.is_vague("water purifier not turning on pls call back")  # "pls call back" is only a sign-off
