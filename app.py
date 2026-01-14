"""
ML Assignment 2 - Streamlit Web Application
Interactive app for classification model evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="ML Classification Models | BITS Pilani",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for modern styling
st.markdown("""
    <style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #ffffff !important;
    }
    
    /* Expander styling - light background */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stExpander"] > div {
        background-color: #ffffff !important;
    }
    
    [data-testid="stExpander"] summary {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    
    [data-testid="stExpander"] p, [data-testid="stExpander"] div, [data-testid="stExpander"] span {
        color: #000000 !important;
    }
    
    /* Dataframe/Table styling - BLACK BACKGROUND WITH WHITE TEXT */
    [data-testid="stDataFrame"] {
        background-color: #000000 !important;
    }
    
    [data-testid="stDataFrame"] > div {
        background-color: #000000 !important;
    }
    
    [data-testid="stDataFrame"] table {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] thead {
        background-color: #1a1a1a !important;
    }
    
    [data-testid="stDataFrame"] th {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    [data-testid="stDataFrame"] tbody {
        background-color: #000000 !important;
    }
    
    [data-testid="stDataFrame"] td {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    [data-testid="stDataFrame"] tr {
        background-color: #000000 !important;
    }
    
    [data-testid="stDataFrame"] tr:nth-child(even) {
        background-color: #1a1a1a !important;
    }
    
    [data-testid="stDataFrame"] tr:nth-child(even) td {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Force white background for entire dataframe container */
    [data-testid="stDataFrameContainer"] {
        background-color: #000000 !important;
    }
    
    [data-testid="stDataFrameContainer"] > div {
        background-color: #000000 !important;
    }
    
    [data-testid="stDataFrameContainer"] table {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stDataFrameContainer"] * {
        background-color: inherit !important;
        color: #ffffff !important;
    }
    
    /* DataFrame icons and buttons - visible with light background */
    [data-testid="stDataFrame"] button {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    [data-testid="stDataFrame"] button:hover {
        background-color: #2a2a2a !important;
    }
    
    [data-testid="stDataFrame"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* DataFrame container buttons */
    [data-testid="stDataFrameContainer"] button {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    [data-testid="stDataFrameContainer"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* All buttons and icons - visible */
    button {
        color: #000000 !important;
    }
    
    button svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* DataFrame toolbar buttons */
    [data-testid="stDataFrame"] [role="button"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] [role="button"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Streamlit dataframe container */
    .stDataFrame {
        background-color: #000000 !important;
    }
    
    .stDataFrame > div {
        background-color: #000000 !important;
    }
    
    /* Force dark theme for all table elements - BLACK BACKGROUND WITH WHITE TEXT */
    table, thead, tbody, tr, th, td {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    table thead {
        background-color: #1a1a1a !important;
    }
    
    table thead th {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    table tbody {
        background-color: #000000 !important;
    }
    
    table tbody tr {
        background-color: #000000 !important;
    }
    
    table tbody tr:nth-child(even) {
        background-color: #1a1a1a !important;
    }
    
    table tbody td {
        background-color: inherit !important;
        color: #000000 !important;
    }
    
    /* Override any inline styles that might set dark backgrounds */
    table[style*="background"],
    thead[style*="background"],
    tbody[style*="background"],
    tr[style*="background"],
    th[style*="background"],
    td[style*="background"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
    }
    
    /* Force text color to black */
    table[style*="color"],
    thead[style*="color"],
    tbody[style*="color"],
    tr[style*="color"],
    th[style*="color"],
    td[style*="color"] {
        color: #000000 !important;
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    /* Subheader */
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Info Box - Light blue with white text */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        color: #1565c0;
    }
    
    .info-box h3, .info-box h4 {
        color: #0d47a1;
        margin-top: 0;
    }
    
    .info-box p, .info-box li {
        color: #1565c0;
    }
    
    /* Success Box - Light green */
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        color: #2e7d32;
    }
    
    .success-box h4 {
        color: #1b5e20;
        margin-top: 0;
    }
    
    .success-box ul, .success-box li {
        color: #2e7d32;
    }
    
    /* Model Card */
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-top: 4px solid #667eea;
    }
    
    /* Section Header */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #212529;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Feature List */
    .feature-item {
        padding: 0.5rem;
        margin: 0.3rem 0;
        background: #f8f9fa;
        border-radius: 5px;
        border-left: 3px solid #667eea;
    }
    
    /* Text color fixes - Force black text */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    p, li, span, div {
        color: #000000 !important;
    }
    
    /* Metric values in black */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    
    /* All text elements in black */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #000000 !important;
    }
    
    /* Metric labels and values */
    .stMetric {
        color: #000000 !important;
    }
    
    .stMetric label {
        color: #000000 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Markdown text color */
    .stMarkdown {
        color: #000000 !important;
    }
    
    /* Metric values in black */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    
    /* Metric labels and values */
    .stMetric label {
        color: #000000 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    
    /* File uploader styling - modern light design */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        border: 2px dashed #667eea !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Inner upload box - increase width and prevent truncation */
    [data-testid="stFileUploader"] > div {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }
    
    [data-testid="stFileUploader"] > div > div {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        padding: 2rem 1.5rem !important;
        box-sizing: border-box !important;
    }
    
    /* Ensure text containers have full width */
    [data-testid="stFileUploader"] > div > div > div {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    
    /* Text elements - prevent truncation */
    [data-testid="stFileUploader"] p {
        color: #495057 !important;
        font-weight: 500 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
        width: auto !important;
        max-width: 100% !important;
    }
    
    /* All text spans and divs in uploader */
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow: visible !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #212529 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stFileUploader"] span {
        color: #495057 !important;
    }
    
    /* Browse files button - modern styling */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    [data-testid="stFileUploader"] button span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stFileUploader"] button div {
        color: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] button p {
        color: #ffffff !important;
    }
    
    /* All text in file uploader button */
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }
    
    /* File uploader drag area text */
    [data-testid="stFileUploader"] > div > div {
        color: #6c757d !important;
    }
    
    [data-testid="stFileUploader"] > div > div > p {
        color: #495057 !important;
        font-size: 1rem !important;
    }
    
    /* File name display - black text, prevent truncation */
    .file-name-display {
        color: #000000 !important;
        font-weight: 500;
        margin-top: 0.5rem;
        padding: 0.75rem 1rem !important;
        background-color: #f8f9fa;
        border-radius: 8px;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        word-wrap: break-word !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    
    .file-name-display strong {
        color: #000000 !important;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .file-name-display {
        display: block !important;
        word-break: break-all !important;
    }
    
    /* Tooltip/Help text styling - white text */
    [data-testid="stTooltip"] {
        color: #ffffff !important;
    }
    
    [data-testid="stTooltip"] p {
        color: #ffffff !important;
    }
    
    [data-testid="stTooltip"] div {
        color: #ffffff !important;
    }
    
    /* Streamlit tooltip content */
    .stTooltip {
        color: #ffffff !important;
    }
    
    .stTooltip p, .stTooltip div, .stTooltip span {
        color: #ffffff !important;
    }
    
    /* Help icon tooltip */
    [data-baseweb="tooltip"] {
        color: #ffffff !important;
    }
    
    [data-baseweb="tooltip"] p {
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] {
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] p {
        color: #ffffff !important;
    }
    
    /* Selectbox/Dropdown styling - white text */
    [data-baseweb="select"] {
        color: #ffffff !important;
    }
    
    [data-baseweb="select"] div {
        color: #ffffff !important;
    }
    
    [data-baseweb="select"] span {
        color: #ffffff !important;
    }
    
    /* Dropdown options text */
    [data-baseweb="menu"] {
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] li {
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] div {
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] span {
        color: #ffffff !important;
    }
    
    /* Selectbox value text */
    [data-baseweb="select"] [aria-selected="true"] {
        color: #ffffff !important;
    }
    
    /* All selectbox related text */
    [data-testid="stSelectbox"] label {
        color: #000000 !important;
    }
    
    [data-testid="stSelectbox"] [data-baseweb="select"] {
        color: #ffffff !important;
    }
    
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        color: #ffffff !important;
    }
    
    [data-testid="stSelectbox"] [data-baseweb="select"] span {
        color: #ffffff !important;
    }
    
    /* Dropdown menu options - white text */
    [data-baseweb="popover"] [role="option"] {
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [role="option"] div {
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [role="option"] span {
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [role="option"] li {
        color: #ffffff !important;
    }
    
    /* Menu items in dropdown */
    [data-baseweb="menu"] [role="option"] {
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] [role="option"] > div {
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] [role="option"] span {
        color: #ffffff !important;
    }
    
    /* All text in dropdown menu - keep white for selectbox dropdown */
    [data-baseweb="popover"] [data-baseweb="menu"] {
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [data-baseweb="menu"] * {
        color: #ffffff !important;
    }
    
    /* Specific targeting for Streamlit selectbox dropdown - keep white text */
    div[data-baseweb="popover"] ul li {
        color: #ffffff !important;
    }
    
    div[data-baseweb="popover"] ul li div {
        color: #ffffff !important;
    }
    
    div[data-baseweb="popover"] ul li span {
        color: #ffffff !important;
    }
    
    /* Popover container - black background */
    div[data-baseweb="popover"] {
        background-color: #000000 !important;
    }
    
    div[data-baseweb="popover"] > div {
        background-color: #000000 !important;
    }
    
    div[data-baseweb="popover"] * {
        color: #ffffff !important;
    }
    
    div[data-baseweb="popover"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Comprehensive styling for all UI elements */
    
    /* Expander content - ensure light background and dark text */
    [data-testid="stExpander"] [data-testid="stDataFrame"] {
        background-color: #000000 !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrame"] > div {
        background-color: #000000 !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrame"] table {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrame"] thead {
        background-color: #1a1a1a !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrame"] th {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrame"] tbody {
        background-color: #000000 !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrame"] td {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrame"] tr {
        background-color: #000000 !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrame"] tr:nth-child(even) {
        background-color: #1a1a1a !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrame"] tr:nth-child(even) td {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Force all table elements in expander to have dark background with white text */
    [data-testid="stExpander"] [data-testid="stDataFrame"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stExpander"] [data-testid="stDataFrameContainer"] {
        background-color: #000000 !important;
    }
    
    /* All icons in expander - white */
    [data-testid="stExpander"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    [data-testid="stExpander"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* DataFrame action buttons (download, search, fullscreen) - dark background with white icons */
    [data-testid="stDataFrame"] [title*="Download"],
    [data-testid="stDataFrame"] [title*="Search"],
    [data-testid="stDataFrame"] [title*="Fullscreen"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] [title*="Download"] svg,
    [data-testid="stDataFrame"] [title*="Search"] svg,
    [data-testid="stDataFrame"] [title*="Fullscreen"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* All SVG icons - default black, but dataframe icons are white */
    svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* DataFrame SVG icons - white */
    [data-testid="stDataFrame"] svg,
    [data-testid="stDataFrameContainer"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Override any dark theme styles */
    [class*="dark"],
    [class*="Dark"],
    [data-theme="dark"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Ensure all text in main content is black */
    .main .block-container * {
        color: #000000 !important;
    }
    
    .main .block-container [data-testid="stDataFrame"] {
        background-color: #000000 !important;
    }
    
    .main .block-container [data-testid="stDataFrame"] * {
        color: #ffffff !important;
    }
    
    /* Value counts and statistics text */
    [data-testid="stText"] p,
    [data-testid="stText"] div,
    [data-testid="stText"] span {
        color: #000000 !important;
    }
    
    /* DataFrame column context menu/popup styling - BLACK BACKGROUND WITH WHITE TEXT */
    [data-baseweb="popover"] {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
    }
    
    [data-baseweb="popover"] [role="menu"],
    [data-baseweb="popover"] [role="menuitem"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [role="menuitem"] div {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [role="menuitem"] span {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [role="menuitem"] p {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Column menu items */
    [data-baseweb="popover"] button {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] button span {
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] button div {
        color: #ffffff !important;
    }
    
    /* All text in popover menus */
    [data-baseweb="popover"] * {
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Menu header buttons in popover */
    [data-baseweb="popover"] [data-baseweb="button"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [data-baseweb="button"] span {
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [data-baseweb="button"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Menu list items - BLACK BACKGROUND WITH WHITE TEXT */
    [data-baseweb="menu"] {
        background-color: #000000 !important;
    }
    
    [data-baseweb="menu"] [role="menuitem"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] [role="menuitem"]:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] [role="menuitem"] span,
    [data-baseweb="menu"] [role="menuitem"] div,
    [data-baseweb="menu"] [role="menuitem"] p {
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] [role="menuitem"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* All elements in menu */
    [data-baseweb="menu"] * {
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Column header menu button */
    [data-testid="stDataFrame"] [role="button"] {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    
    [data-testid="stDataFrame"] th [role="button"] {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] th [role="button"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* All popover content */
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    div[data-baseweb="popover"] * {
        color: #000000 !important;
    }
    
    /* Matplotlib/Plot fullscreen button - light background with visible icon */
    [data-testid="stImage"] button,
    [data-testid="stImage"] [role="button"],
    .stImage button,
    .stImage [role="button"] {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    [data-testid="stImage"] button svg,
    [data-testid="stImage"] [role="button"] svg,
    .stImage button svg,
    .stImage [role="button"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Plot container buttons */
    [data-testid="stImage"] [title*="Fullscreen"],
    [data-testid="stImage"] [title*="fullscreen"],
    .stImage [title*="Fullscreen"],
    .stImage [title*="fullscreen"] {
        background-color: #f8f9fa !important;
        color: #000000 !important;
        border: 1px solid #dee2e6 !important;
    }
    
    [data-testid="stImage"] [title*="Fullscreen"] svg,
    [data-testid="stImage"] [title*="fullscreen"] svg,
    .stImage [title*="Fullscreen"] svg,
    .stImage [title*="fullscreen"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* All buttons in image/plot containers */
    div[data-testid="stImage"] button,
    div[data-testid="stImage"] [role="button"] {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    div[data-testid="stImage"] button svg,
    div[data-testid="stImage"] [role="button"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Pyplot figure buttons */
    .stPlotlyChart button,
    figure button,
    .matplotlib-container button {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    .stPlotlyChart button svg,
    figure button svg,
    .matplotlib-container button svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* More specific targeting for plotly/matplotlib fullscreen buttons */
    [data-testid="stImage"] > div button,
    [data-testid="stImage"] > div [role="button"],
    [data-testid="stImage"] > div > div button {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    [data-testid="stImage"] > div button svg,
    [data-testid="stImage"] > div [role="button"] svg,
    [data-testid="stImage"] > div > div button svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Target all buttons near plots/images */
    .element-container [data-testid="stImage"] button,
    .element-container [data-testid="stImage"] [role="button"],
    .stImageContainer button,
    .stImageContainer [role="button"] {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    .element-container [data-testid="stImage"] button svg,
    .element-container [data-testid="stImage"] [role="button"] svg,
    .stImageContainer button svg,
    .stImageContainer [role="button"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Universal button styling in plot areas */
    [data-testid="stImage"] * button,
    [data-testid="stImage"] * [role="button"] {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    [data-testid="stImage"] * button svg,
    [data-testid="stImage"] * [role="button"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Force light background for any button in plot containers */
    div:has([data-testid="stImage"]) button,
    div:has([data-testid="stImage"]) [role="button"] {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    div:has([data-testid="stImage"]) button svg,
    div:has([data-testid="stImage"]) [role="button"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Additional targeting for plot toolbar buttons */
    [data-testid="stImage"] ~ button,
    [data-testid="stImage"] + button,
    [data-testid="stImage"] + div button {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    [data-testid="stImage"] ~ button svg,
    [data-testid="stImage"] + button svg,
    [data-testid="stImage"] + div button svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Target buttons that are siblings or children of plot containers */
    .element-container:has([data-testid="stImage"]) button,
    .element-container:has([data-testid="stImage"]) [role="button"],
    .element-container:has([data-testid="stImage"]) > div > button {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    .element-container:has([data-testid="stImage"]) button svg,
    .element-container:has([data-testid="stImage"]) [role="button"] svg,
    .element-container:has([data-testid="stImage"]) > div > button svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Force all buttons in the plot area to have light background */
    [class*="plot"] button,
    [class*="Plot"] button,
    [class*="chart"] button,
    [class*="Chart"] button,
    [class*="figure"] button,
    [class*="Figure"] button {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    [class*="plot"] button svg,
    [class*="Plot"] button svg,
    [class*="chart"] button svg,
    [class*="Chart"] button svg,
    [class*="figure"] button svg,
    [class*="Figure"] button svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Aggressive targeting for all buttons - override any dark backgrounds */
    button[style*="background"],
    button[style*="Background"],
    [role="button"][style*="background"],
    [role="button"][style*="Background"] {
        background-color: #f8f9fa !important;
        background: #f8f9fa !important;
    }
    
    /* Target buttons with dark backgrounds specifically */
    button[style*="rgb(0, 0, 0)"],
    button[style*="rgb(0,0,0)"],
    button[style*="#000"],
    button[style*="#000000"],
    [role="button"][style*="rgb(0, 0, 0)"],
    [role="button"][style*="#000"] {
        background-color: #f8f9fa !important;
        background: #f8f9fa !important;
    }
    
    /* Force light background on any button near plots */
    [data-testid="stImage"] + * button,
    [data-testid="stImage"] ~ * button,
    [data-testid="stImage"] ~ * [role="button"] {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    [data-testid="stImage"] + * button svg,
    [data-testid="stImage"] ~ * button svg,
    [data-testid="stImage"] ~ * [role="button"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Universal button fix for plot areas */
    .main .block-container button:not([data-baseweb="select"] button):not([data-testid="stFileUploader"] button) {
        background-color: #f8f9fa !important;
    }
    
    .main .block-container button:not([data-baseweb="select"] button):not([data-testid="stFileUploader"] button) svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    """Load a saved model"""
    possible_paths = [
        f'saved_models/{model_name}.pkl',
        os.path.join(os.path.dirname(__file__), 'saved_models', f'{model_name}.pkl')
    ]

    for model_path in possible_paths:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    return None

@st.cache_resource
def load_scaler():
    """Load the scaler"""
    possible_paths = [
        'saved_models/scaler.pkl',
        os.path.join(os.path.dirname(__file__), 'saved_models', 'scaler.pkl')
    ]
    
    for scaler_path in possible_paths:
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
    return None

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    try:
        # Handle multiclass classification
        if y_pred_proba.ndim == 1:
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            # Binary classification
            metrics = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'AUC': roc_auc_score(y_true, y_pred_proba[:, 1]),
                'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'F1': f1_score(y_true, y_pred, average='binary', zero_division=0),
                'MCC': matthews_corrcoef(y_true, y_pred)
            }
        else:
            # Multiclass classification
            metrics = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'AUC': roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'),
                'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'MCC': matthews_corrcoef(y_true, y_pred)
            }
        return metrics
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix with better styling"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique class labels
    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_labels = [f'Class {int(c)}' for c in classes]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application"""
    
    # Enhanced Header with gradient
    st.markdown('<h1 class="main-header">🎓 ML Classification Models Evaluation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">BITS Pilani | M.Tech (AIML/DSE) | Machine Learning Assignment 2</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        st.markdown("---")
        page = st.radio("", ["📊 Model Evaluation", "ℹ️ About"], label_visibility="collapsed")
        
        if page == "📊 Model Evaluation":
            st.markdown("### 🤖 Model Selection")
            model_options = {
                "📈 Logistic Regression": "logistic_regression",
                "🌳 Decision Tree": "decision_tree",
                "🔍 K-Nearest Neighbor": "knn",
                "📊 Naive Bayes": "naive_bayes",
                "🌲 Random Forest": "random_forest",
                "⚡ XGBoost": "xgboost"
            }
            
            selected_model_name = st.selectbox(
                "Choose a model",
                list(model_options.keys()),
                help="Select the machine learning model to evaluate"
            )
            
            selected_model_key = model_options[selected_model_name]
            
            st.markdown("---")
            st.markdown("### 📁 Dataset Upload")
            uploaded_file = st.file_uploader(
                "Upload test data (CSV)",
                type=['csv'],
                help="Upload your test dataset CSV file for model evaluation"
            )
            
            # Display file name in black text below uploader
            if uploaded_file is not None:
                file_size = len(uploaded_file.getvalue()) / 1024  # Size in KB
                file_size_str = f"{file_size:.1f}KB" if file_size < 1024 else f"{file_size/1024:.1f}MB"
                st.markdown(
                    f'<div class="file-name-display">'
                    f'📄 <strong>Selected File:</strong> <span style="word-break: break-all; display: inline-block; max-width: 100%;">{uploaded_file.name}</span><br>'
                    f'<span style="font-size: 0.9em; color: #6c757d; margin-top: 0.25rem; display: block;">Size: {file_size_str}</span>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            st.markdown("### 📝 Quick Info")
            st.info("""
            **Requirements:**
            - CSV file format
            - Must have 'target' column
            - Features should match training data
            """)
    
    if page == "📊 Model Evaluation":
        # Main content area
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ **Dataset loaded successfully!** Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
                
                # Check if target column exists
                target_col = None
                for col in ['target', 'Target', 'TARGET', 'exam_score', 'GradeClass']:
                    if col in df.columns:
                        target_col = col
                        break
                
                if target_col is None:
                    st.error("❌ **Error:** Target column not found!")
                    st.info("Please ensure your CSV file has a 'target' column (or 'exam_score', 'GradeClass') for the labels.")
                    return
                
                # Rename target column if needed
                if target_col != 'target':
                    df = df.rename(columns={target_col: 'target'})
                
                # Display dataset info in expander
                with st.expander("📊 Dataset Preview & Information", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**First 10 rows:**")
                        st.dataframe(df.head(10), use_container_width=True)
                    with col2:
                        st.write("**Dataset Statistics:**")
                        st.write(f"- **Total Rows:** {df.shape[0]:,}")
                        st.write(f"- **Total Columns:** {df.shape[1]}")
                        st.write(f"- **Features:** {df.shape[1] - 1}")
                        st.write(f"- **Target Distribution:**")
                        st.write(df['target'].value_counts())
                
                # Separate features and target
                X = df.drop('target', axis=1)
                y = df['target']
                
                # Load model and scaler
                model = load_model(selected_model_key)
                scaler = load_scaler()
                
                if model is None:
                    st.error(f"❌ **Model '{selected_model_name}' not found!**")
                    st.info("Please ensure models are trained and saved in the 'saved_models' directory.")
                    return
                
                # Scale features if needed
                models_requiring_scaling = ['logistic_regression', 'knn', 'naive_bayes']
                if selected_model_key in models_requiring_scaling:
                    if scaler is None:
                        st.error("❌ Scaler not found!")
                        return
                    X_scaled = scaler.transform(X)
                    y_pred = model.predict(X_scaled)
                    y_pred_proba = model.predict_proba(X_scaled)
                else:
                    y_pred = model.predict(X)
                    y_pred_proba = model.predict_proba(X)
                
                # Calculate metrics
                metrics = calculate_metrics(y, y_pred, y_pred_proba)
                
                if metrics:
                    # Display metrics with enhanced styling
                    st.markdown('<div class="section-header">📈 Evaluation Metrics</div>', unsafe_allow_html=True)
                    
                    # Create columns for metrics display with better spacing
                    col1, col2, col3 = st.columns(3)
                    col4, col5, col6 = st.columns(3)
                    
                    # Color-coded metrics
                    metric_colors = {
                        'Accuracy': '#4caf50',
                        'AUC': '#2196f3',
                        'Precision': '#ff9800',
                        'Recall': '#9c27b0',
                        'F1': '#f44336',
                        'MCC': '#00bcd4'
                    }
                    
                    with col1:
                        st.metric("🎯 Accuracy", f"{metrics['Accuracy']:.4f}")
                    with col2:
                        st.metric("📊 AUC Score", f"{metrics['AUC']:.4f}")
                    with col3:
                        st.metric("🎪 Precision", f"{metrics['Precision']:.4f}")
                    with col4:
                        st.metric("🔍 Recall", f"{metrics['Recall']:.4f}")
                    with col5:
                        st.metric("⚖️ F1 Score", f"{metrics['F1']:.4f}")
                    with col6:
                        st.metric("📐 MCC Score", f"{metrics['MCC']:.4f}")
                    
                    # Metrics table
                    st.markdown("### 📋 Detailed Metrics Table")
                    metrics_df = pd.DataFrame([metrics]).T
                    metrics_df.columns = ['Value']
                    metrics_df['Percentage'] = (metrics_df['Value'] * 100).round(2).astype(str) + '%'
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Confusion Matrix
                    st.markdown('<div class="section-header">🔍 Confusion Matrix</div>', unsafe_allow_html=True)
                    fig = plot_confusion_matrix(y, y_pred, selected_model_name)
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.markdown('<div class="section-header">📊 Classification Report</div>', unsafe_allow_html=True)
                    from sklearn.metrics import classification_report
                    report = classification_report(y, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ **Error processing file:** {str(e)}")
                st.info("Please ensure your CSV file is properly formatted and contains the required columns.")
        else:
            # Enhanced default view
            st.info("👆 **Get Started:** Upload a CSV file from the sidebar to begin model evaluation.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 📋 Step-by-Step Instructions:
                1. **Select Model**: Choose a model from the dropdown in the sidebar
                2. **Upload Dataset**: Click "Browse files" and select your test CSV file
                3. **View Results**: See evaluation metrics, confusion matrix, and classification report
                
                ### ✅ Dataset Requirements:
                - CSV file format
                - Must contain a 'target' column (or 'exam_score', 'GradeClass')
                - Feature columns should match the training data structure
                - Target can be binary or multi-class
                """)
            
            with col2:
                st.markdown("""
                ### 🎯 Available Models:
                - **Logistic Regression**: Linear classification
                - **Decision Tree**: Tree-based model
                - **K-Nearest Neighbor**: Instance-based learning
                - **Naive Bayes**: Probabilistic classifier
                - **Random Forest**: Ensemble method
                - **XGBoost**: Gradient boosting
                
                ### 📊 Evaluation Metrics:
                All models are evaluated using 6 metrics:
                - Accuracy, AUC, Precision, Recall, F1, MCC
                """)
    
    elif page == "ℹ️ About":
        st.markdown('<div class="section-header">📚 About This Application</div>', unsafe_allow_html=True)
        
        st.info("""
        **About This Application**
        
        This application demonstrates the evaluation of multiple machine learning classification models 
        for student performance prediction. Built as part of **BITS Pilani M.Tech (AIML/DSE) 
        Machine Learning Assignment 2**.
        """)
        
        # Models Section
        st.markdown("### 🤖 Machine Learning Models Implemented")
        
        models_info = [
            {
                "name": "📈 Logistic Regression",
                "description": "Linear classification model that uses logistic function to model probabilities. Best for linearly separable data.",
                "pros": "Interpretable, fast, provides probability estimates",
                "cons": "Assumes linear relationship"
            },
            {
                "name": "🌳 Decision Tree",
                "description": "Tree-based model that splits data based on feature values. Highly interpretable and can capture non-linear relationships.",
                "pros": "No feature scaling needed, interpretable, handles non-linear data",
                "cons": "Prone to overfitting"
            },
            {
                "name": "🔍 K-Nearest Neighbor (KNN)",
                "description": "Instance-based learning algorithm that classifies based on similarity to k nearest neighbors.",
                "pros": "Simple, no training phase, adapts to new data",
                "cons": "Computationally expensive, sensitive to irrelevant features"
            },
            {
                "name": "📊 Naive Bayes",
                "description": "Probabilistic classifier based on Bayes' theorem with strong independence assumptions.",
                "pros": "Fast, works well with small datasets, handles multiple classes",
                "cons": "Assumes feature independence"
            },
            {
                "name": "🌲 Random Forest",
                "description": "Ensemble method combining multiple decision trees using bagging. Reduces overfitting.",
                "pros": "Robust, handles non-linear data, provides feature importance",
                "cons": "Less interpretable than single tree"
            },
            {
                "name": "⚡ XGBoost",
                "description": "Gradient boosting ensemble method that sequentially builds models to correct errors.",
                "pros": "High performance, handles missing values, feature importance",
                "cons": "Requires hyperparameter tuning, computationally intensive"
            }
        ]
        
        for model in models_info:
            with st.expander(model["name"], expanded=False):
                st.write(f"**Description:** {model['description']}")
                st.write(f"**✅ Advantages:** {model['pros']}")
                st.write(f"**⚠️ Limitations:** {model['cons']}")
        
        # Metrics Section
        st.markdown("### 📊 Evaluation Metrics Explained")
        
        metrics_info = {
            "🎯 Accuracy": "Proportion of correct predictions out of all predictions. Simple but can be misleading with imbalanced datasets.",
            "📊 AUC Score": "Area Under the ROC Curve. Measures model's ability to distinguish between classes. Higher is better (max 1.0).",
            "🎪 Precision": "Proportion of positive predictions that are actually positive. Measures model's reliability.",
            "🔍 Recall": "Proportion of actual positives that are correctly identified. Measures model's completeness.",
            "⚖️ F1 Score": "Harmonic mean of precision and recall. Balances both metrics. Best when you need a single metric.",
            "📐 MCC Score": "Matthews Correlation Coefficient. Balanced measure considering all confusion matrix values. Range: -1 to +1."
        }
        
        for metric, description in metrics_info.items():
            st.markdown(f"**{metric}**")
            st.write(description)
            st.markdown("---")
        
        # Dataset Section
        st.markdown("### 📁 Dataset Information")
        
        st.success("""
        **Student Performance Dataset**
        
        - **Source:** Student_performance_data.csv
        - **Instances:** 2,393 samples
        - **Features:** 14 features including:
          - Demographics: Age, Gender, Ethnicity
          - Academic: Study Time, Absences, GPA
          - Support: Parental Education, Tutoring, Parental Support
          - Activities: Extracurricular, Sports, Music, Volunteering
        - **Target:** GradeClass (Multi-class classification)
        - **Preprocessing:** Categorical encoding, feature scaling for certain models
        """)
        
        # Technical Details
        st.markdown("### 🔧 Technical Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Technologies Used:**
            - Python 3.x
            - Streamlit (Web Framework)
            - Scikit-learn (ML Models)
            - XGBoost (Gradient Boosting)
            - Pandas & NumPy (Data Processing)
            - Matplotlib & Seaborn (Visualization)
            """)
        
        with col2:
            st.markdown("""
            **Model Training:**
            - Train-Test Split: 80-20
            - Stratified sampling for balanced classes
            - Feature scaling for Logistic Regression, KNN, Naive Bayes
            - Hyperparameters optimized for each model
            """)
        
        # Usage Instructions
        st.markdown("### 🚀 How to Use")
        
        st.markdown("""
        1. **Navigate to Model Evaluation** page from the sidebar
        2. **Select a model** from the dropdown menu
        3. **Upload your test dataset** (CSV file with 'target' column)
        4. **View results** including:
           - All 6 evaluation metrics
           - Confusion matrix visualization
           - Detailed classification report
        5. **Compare models** by switching between different models
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6c757d; padding: 2rem;">
        <p><strong>BITS Pilani</strong> | Work Integrated Learning Programmes Division</p>
        <p>M.Tech (AIML/DSE) | Machine Learning Assignment 2</p>
        <p style="font-size: 0.9rem;">Built with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()