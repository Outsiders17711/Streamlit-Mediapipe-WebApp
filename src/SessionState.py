"""per-session state backed by st.session_state."""

import streamlit as st

__all__ = ["getSessionState"]


def getSessionState(**kwargs):
    """return the current session state, seeding defaults on first use."""
    for key, val in kwargs.items():
        if key not in st.session_state:
            st.session_state[key] = val
    return st.session_state
