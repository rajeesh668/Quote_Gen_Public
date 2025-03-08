import gradio as gr
import pandas as pd
import re
import os
import pyperclip  # Ensure you have pyperclip installed

# ---------------------
# Data Loading Functions
# ---------------------
def load_data():
    """
    Loads Models.csv and SKU.csv from the local 'data' folder.
    """
    try:
        models_df = pd.read_csv("data/Models.csv", encoding="ISO-8859-1")
        sku_df = pd.read_csv("data/SKU.csv", encoding="ISO-8859-1")
        return models_df, sku_df
    except Exception as e:
        print("Error loading CSV files:", e)
        return pd.DataFrame(), pd.DataFrame()

models_df, sku_df = load_data()

def load_license_data():
    """
    Loads License.csv from the local 'data' folder.
    """
    try:
        license_df = pd.read_csv("data/License.csv", encoding="ISO-8859-1")
        return license_df
    except Exception as e:
        print("Error loading License CSV:", e)
        return pd.DataFrame()

def load_fixed_skus():
    """
    Loads FixedSKUs.csv from the local 'data' folder.
    """
    try:
        fixed_df = pd.read_csv("data/FixedSKUs.csv", encoding="ISO-8859-1")
        fixed_dict = dict(zip(fixed_df["Model"].str.strip(), fixed_df["SKU"].str.strip()))
        return fixed_dict
    except Exception as e:
        print("Error loading Fixed SKUs CSV:", e)
        return {}

fixed_skus = load_fixed_skus()

# ---------------------
# Discount Tables
# ---------------------
partner_discounts = {  # for SMB
    "Authorised": 0.14,
    "Silver": 0.19,
    "Gold": 0.24,
    "Platinum": 0.29
}
partner_discounts_DR = {  # for NEW MME when deal registration is checked
    "Authorised": 0.20,
    "Silver": 0.25,
    "Gold": 0.30,
    "Platinum": 0.35
}
partner_incumbency = {  # for RENEWAL MME when incumbency is checked
    "Authorised": 0.10,
    "Silver": 0.15,
    "Gold": 0.20,
    "Platinum": 0.25
}

# ---------------------
# Helper Functions using gr.State instead of a global variable
# ---------------------
def remove_last_item(quote_state):
    """
    Removes the last item from the quote state list.
    """
    state = quote_state if quote_state is not None else []
    if state:
        state.pop()
    df = pd.DataFrame(state, columns=[
        "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
        "Reseller Price (USD)", "Disc. Price (USD)",
        "Override Discount (%)", "Additional Discount (%)",
        "Classification", "Product Type", "Status"
    ])
    return df, state

def reset_quote(quote_state):
    """
    Resets the quote state to an empty list.
    """
    state = []
    df = pd.DataFrame(columns=[
        "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
        "Reseller Price (USD)", "Disc. Price (USD)",
        "Override Discount (%)", "Additional Discount (%)",
        "Classification", "Product Type", "Status"
    ])
    return df, state

def update_models(category):
    """
    Updates the 'Model' dropdown based on the selected category.
    """
    if not models_df.empty and category in models_df.columns:
        models_list = models_df[category].dropna().tolist()
        return gr.update(choices=models_list, value=models_list[0] if models_list else "")
    return gr.update(choices=[], value="")

def update_license_toggle(category):
    """
    Shows/hides the firewall license toggle if the category is 'Firewall'.
    """
    if str(category).lower() == "firewall":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False, value=False)

def update_license_box(use_license, category):
    """
    Updates the license dropdown if the user enables license for firewalls.
    """
    if str(category).lower() == "firewall" and use_license:
        license_df = load_license_data()
        if not license_df.empty:
            if "License" in license_df.columns:
                licenses = license_df["License"].dropna().tolist()
            else:
                licenses = license_df.iloc[:, 0].dropna().tolist()
            return gr.update(choices=licenses, visible=True, value=licenses[0] if licenses else "")
        else:
            return gr.update(choices=[], visible=False)
    else:
        return gr.update(visible=False)

# ---------------------
# New: MDR Onboarding Display
# ---------------------
def update_category_message(selected_category):
    """
    Shows a message about MDR Onboarding if the category is 'MDR'.
    """
    if selected_category.strip().lower() in ["m d r", "mdr"]:
        message = """
        <p style="color:red; font-weight:bold; text-align:center;">
        Suggested to add MDR Onboarding for NEW business<br>
        Add <span style="color:green;">PRPE0A00ZZPCAA</span> up to 1000 users<br>
        Add <span style="color:green;">PRPN0A00ZZPCAA</span> more than 1000 users
        </p>
        """
        return gr.update(visible=True, value=message)
    else:
        return gr.update(visible=False)

# ---------------------
# New: Switch Support License Toggle Functions
# ---------------------
def update_switch_support_toggle(category):
    """
    Shows/hides the switch support license checkbox if category is 'Switches'.
    """
    if str(category).strip().lower() == "switches":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False, value=False)

def find_matching_sku(category, model_selected, transaction_type, term, quantity, manual_sku="", license_selected=""):
    """
    Finds the matching SKU in the SKU DataFrame based on various criteria.
    """
    # Manual SKU override
    if manual_sku.strip():
        sku_details = sku_df[sku_df["SKU"].str.lower() == manual_sku.strip().lower()]
        if not sku_details.empty:
            row = sku_details.iloc[0]
            return row["SKU"], row["Description"], float(row["Price"]), row["Product Type"], term
        else:
            return manual_sku.strip(), "Manual SKU Not Found", 0.0, "Unknown", term

    # Check fixed SKUs if not using a firewall license
    if not (str(category).lower() == "firewall" and license_selected):
        if model_selected in fixed_skus:
            sku = fixed_skus[model_selected]
            sku_details = sku_df[sku_df["SKU"] == sku]
            if not sku_details.empty:
                description = sku_details.iloc[0]["Description"]
                price = float(sku_details.iloc[0]["Price"])
                product_type = sku_details.iloc[0]["Product Type"]
                return sku, description, price, product_type, "-"

    # For firewall + license
    if str(category).lower() == "firewall" and license_selected:
        search_str = f"{model_selected.strip()} {license_selected.strip()}"
    else:
        search_str = model_selected.strip()

    combined_pattern = re.escape(search_str)
    transaction_type = transaction_type if isinstance(transaction_type, str) else ""
    term_str = str(term)

    filtered_sku = sku_df[
        (sku_df["Description"].str.contains(combined_pattern, na=False, case=False, regex=True)) &
        (sku_df["Transaction Type"].str.contains(transaction_type, na=False, case=False)) &
        (sku_df["Description"].str.contains(term_str, na=False))
    ]

    # Exclude hardware/appliance if it's a firewall license
    if str(category).lower() == "firewall" and license_selected:
        filtered_sku = filtered_sku[~filtered_sku["Product Type"].isin(["Hardware", "Appliance"])]

    # Attempt to match user/server ranges in the SKU description
    for _, row in filtered_sku.iterrows():
        desc = row["Description"].lower()
        match_users_servers = re.search(r"(\d+)-(\d+)\s+users and servers", desc)
        if match_users_servers:
            min_qty, max_qty = map(int, match_users_servers.groups())
            if min_qty <= quantity <= max_qty:
                return row["SKU"], row["Description"], float(row["Price"]), row["Product Type"], term

        match_users = re.search(r"(\d+)-(\d+)\s+users(?! and servers)", desc)
        if match_users:
            min_qty, max_qty = map(int, match_users.groups())
            if min_qty <= quantity <= max_qty:
                return row["SKU"], row["Description"], float(row["Price"]), row["Product Type"], term

        match_servers = re.search(r"(\d+)-(\d+)\s+servers", desc)
        if match_servers:
            min_qty, max_qty = map(int, match_servers.groups())
            if min_qty <= quantity <= max_qty:
                return row["SKU"], row["Description"], float(row["Price"]), row["Product Type"], term

    # If no specific range match, return the first found
    if not filtered_sku.empty:
        row = filtered_sku.iloc[0]
        return row["SKU"], row["Description"], float(row["Price"]), row["Product Type"], term

    return "", "No matching SKU found", 0.0, "", term

# ---------------------
# New: Wireless License Support SKU Finder
# ---------------------
def find_wireless_license_support_sku(model_name, transaction_type, term):
    """
    Finds the wireless support SKU for AP6 access points.
    """
    if transaction_type == "Renewal":
        search_str = f"Access Points Support for {model_name.strip()} - {term} - Renewal"
    else:
        search_str = f"Access Points Support for {model_name.strip()} - {term}"

    pattern = re.escape(search_str)
    filtered = sku_df[sku_df["Description"].str.contains(pattern, na=False, case=False, regex=True)]
    if not filtered.empty:
        row = filtered.iloc[0]
        return row["SKU"], row["Description"], float(row["Price"]), row["Product Type"]
    return "", "No matching SKU found for wireless license support", 0.0, ""

# ---------------------
# New: Switch Support License SKU Finder
# ---------------------
def find_switch_support_sku(model_name, transaction_type, term):
    """
    Finds the switch support SKU for a given model.
    """
    if transaction_type == "Renewal":
        search_str = f"Switch Support and Services for {model_name.strip()} - {term} - Renewal"
    else:
        search_str = f"Switch Support and Services for {model_name.strip()} - {term}"

    pattern = re.escape(search_str)
    filtered = sku_df[sku_df["Description"].str.contains(pattern, na=False, case=False, regex=True)]
    if not filtered.empty:
        row = filtered.iloc[0]
        return row["SKU"], row["Description"], float(row["Price"]), row["Product Type"]
    return "", "No matching SKU found for switch support", 0.0, ""

# ---------------------
# Modified Add Line Item Function (Now with state)
# ---------------------
def add_line_item(
    category, model_selected, transaction_type, term, quantity, partner_type, manual_sku,
    use_license, license_selected, deal_registered, incumbency, use_wireless_license,
    use_switch_support, quote_state
):
    """
    Adds a new line item to the quote (stored in gr.State).
    """
    state = quote_state if quote_state is not None else []

    # Branch for Switches + Support
    if str(category).strip().lower() == "switches" and use_switch_support:
        sw_sku, sw_desc, sw_price, sw_type = find_switch_support_sku(model_selected, transaction_type, term)
        if sw_sku:
            state.append([
                sw_sku, sw_desc, term, 1, f"{sw_price:.2f}",
                f"{sw_price:.2f}", f"{sw_price:.2f}",
                "0", "0", "Switch Support", sw_type, ""
            ])
            df = pd.DataFrame(state, columns=[
                "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
                "Reseller Price (USD)", "Disc. Price (USD)",
                "Override Discount (%)", "Additional Discount (%)",
                "Classification", "Product Type", "Status"
            ])
            return df, state

    # Branch for Wireless Licenses (AP6)
    if use_wireless_license and model_selected.strip().upper().startswith("AP6"):
        lic_sku, lic_desc, lic_price, lic_type = find_wireless_license_support_sku(
            model_selected, transaction_type, term
        )
        if lic_sku:
            discount_rate = 0.10
            reseller_price = lic_price * (1 - discount_rate)
            disc_price = reseller_price
            state.append([
                lic_sku, lic_desc, term, 1, f"{lic_price:.2f}",
                f"{reseller_price:.2f}", f"{disc_price:.2f}",
                f"{discount_rate*100:.0f}", "0", "Wireless License", lic_type, ""
            ])
            df = pd.DataFrame(state, columns=[
                "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
                "Reseller Price (USD)", "Disc. Price (USD)",
                "Override Discount (%)", "Additional Discount (%)",
                "Classification", "Product Type", "Status"
            ])
            return df, state

    # Normal SKU Lookup
    if not use_license:
        license_selected = ""

    sku, description, original_price_usd, product_type, adjusted_term = find_matching_sku(
        category, model_selected, transaction_type, term, quantity, manual_sku, license_selected
    )
    if product_type.strip().lower() in ["hardware", "appliance", "proservices"]:
        adjusted_term = "-"

    if not sku:
        # No match found
        df = pd.DataFrame(state, columns=[
            "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
            "Reseller Price (USD)", "Disc. Price (USD)",
            "Override Discount (%)", "Additional Discount (%)",
            "Classification", "Product Type", "Status"
        ])
        return df, state

    discount_rate = 0.10
    sku_rows = sku_df[sku_df["SKU"] == sku]
    classification = "-"
    status_text = ""

    # Classification-based discounts
    if not sku_rows.empty:
        val = sku_rows.iloc[0].get("Classification", "")
        raw_class = str(val).strip().upper()
        if raw_class in ["SMB", "MME"]:
            classification = raw_class
        else:
            classification = "-"

        if classification in ["SMB", "MME"]:
            if transaction_type == "New":
                if classification == "MME":
                    if deal_registered:
                        discount_rate = partner_discounts_DR.get(partner_type, 0)
                        status_text = f"{classification}, DR: {discount_rate*100:.0f}%"
                    else:
                        discount_rate = 0.10
                        status_text = f"{classification}, No DR: 10%"
                else:
                    discount_rate = partner_discounts.get(partner_type, 0)
                    status_text = f"{classification}, {discount_rate*100:.0f}%"
            elif transaction_type == "Renewal":
                if classification == "MME":
                    if incumbency:
                        discount_rate = partner_incumbency.get(partner_type, 0)
                        status_text = f"{classification}, Incumbency: {discount_rate*100:.0f}%"
                    else:
                        discount_rate = 0.10
                        status_text = f"{classification}, No Incumbency: 10%"
                else:
                    discount_rate = partner_discounts.get(partner_type, 0)
                    status_text = f"{classification}, {discount_rate*100:.0f}%"
            else:
                discount_rate = 0.10
                status_text = f"{classification}, 10%"
        else:
            discount_rate = 0.10
            status_text = f"Non-Core, 10%"

    reseller_price_usd = original_price_usd * (1 - discount_rate)
    disc_price_usd = reseller_price_usd

    state.append([
        sku, description, adjusted_term, quantity, f"{original_price_usd:.2f}",
        f"{reseller_price_usd:.2f}", f"{disc_price_usd:.2f}",
        f"{discount_rate*100:.0f}", "0",
        classification, product_type, status_text
    ])
    df = pd.DataFrame(state, columns=[
        "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
        "Reseller Price (USD)", "Disc. Price (USD)",
        "Override Discount (%)", "Additional Discount (%)",
        "Classification", "Product Type", "Status"
    ])
    return df, state

def recalc_boQ(boq_df):
    """
    Recalculates BOQ after the user edits override or additional discounts.
    """
    new_rows = []
    for idx, row in boq_df.iterrows():
        try:
            original = float(row["Original Price (USD)"])
        except:
            original = 0.0
        try:
            override_disc = float(row.get("Override Discount (%)", 0))
        except:
            override_disc = 0.0
        try:
            additional_disc = float(row.get("Additional Discount (%)", 0))
        except:
            additional_disc = 0.0

        final_price = original * (1 - override_disc/100) * (1 - additional_disc/100)
        row["Disc. Price (USD)"] = f"{final_price:.2f}"
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    new_df = new_df[[
        "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
        "Reseller Price (USD)", "Disc. Price (USD)",
        "Override Discount (%)", "Additional Discount (%)",
        "Classification", "Product Type", "Status"
    ]]
    return new_df

def generate_final_quote(shipping_cost, boq_data):
    """
    Generates the final quote in SAR with shipping and VAT calculations.
    """
    if shipping_cost is None or shipping_cost == "":
        shipping_cost = 4

    if boq_data is None or boq_data.empty:
        return pd.DataFrame(columns=[
            "SKU", "Description", "Term", "Quantity", 
            "Unit Price (SAR)", "Total Price (SAR)"
        ])

    df_quote = boq_data.copy()

    # Convert USD to SAR
    df_quote["Unit Price (SAR)"] = df_quote["Disc. Price (USD)"].astype(float) * 3.7575
    df_quote["Total Price (SAR)"] = df_quote["Unit Price (SAR)"] * df_quote["Quantity"].astype(float)

    total_subtotal_sar = df_quote["Total Price (SAR)"].sum()

    # Calculate shipping only for Appliance/Hardware
    shipping_items = df_quote[df_quote["Product Type"].isin(["Appliance", "Hardware"])]
    total_original_usd = shipping_items.apply(
        lambda r: float(r["Original Price (USD)"]) * float(r["Quantity"]), axis=1
    ).sum()
    total_shipping_sar = total_original_usd * (float(shipping_cost) / 100.0) * 3.7575

    total_vat_sar = 0.15 * (total_subtotal_sar + total_shipping_sar)
    grand_total_sar = total_subtotal_sar + total_shipping_sar + total_vat_sar

    df_final_quote = df_quote[[
        "SKU", "Description", "Term", "Quantity", 
        "Unit Price (SAR)", "Total Price (SAR)"
    ]]

    df_summary = pd.DataFrame([
        ["", "Shipping Cost", "", "", "", f"{total_shipping_sar:.2f}"],
        ["", "VAT (15%)", "", "", "", f"{total_vat_sar:.2f}"],
        ["", "GRAND TOTAL", "", "", "", f"{grand_total_sar:.2f}"]
    ], columns=df_final_quote.columns)

    df_final = pd.concat([df_final_quote, df_summary], ignore_index=True)
    return df_final

def toggle_shipping_box(override_value):
    """
    Shows/hides the shipping cost input if the user wants to override it.
    """
    if override_value:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False, value=4)

def download_final_quote(boq_data):
    """
    Saves the final quote as an Excel file named 'Final_Quote.xlsx' 
    in the current working directory and returns the file path.
    """
    if boq_data is None or boq_data.empty:
        return None

    df_final = generate_final_quote(4, boq_data)
    file_path = "Final_Quote.xlsx"  # Save in current directory

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df_final.to_excel(writer, index=False, sheet_name="Quote", startrow=0)
        workbook = writer.book
        worksheet = writer.sheets["Quote"]

        header_format = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#4F81BD",
            "align": "center",
            "valign": "vcenter",
            "border": 1
        })
        data_format = workbook.add_format({
            "border": 1,
            "align": "center",
            "valign": "vcenter"
        })
        price_format = workbook.add_format({
            "border": 1,
            "num_format": "#,##0.00",
            "align": "right",
            "valign": "vcenter"
        })
        wrap_text_format = workbook.add_format({
            "border": 1,
            "align": "left",
            "valign": "vcenter",
            "text_wrap": True
        })
        summary_label_format = workbook.add_format({
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "border": 1
        })
        summary_value_format = workbook.add_format({
            "bold": True,
            "num_format": "#,##0.00",
            "align": "right",
            "valign": "vcenter",
            "border": 1
        })
        grand_total_format = workbook.add_format({
            "bold": True,
            "font_size": 12,
            "bg_color": "#D9E1F2",
            "num_format": "#,##0.00",
            "align": "right",
            "valign": "vcenter",
            "border": 1
        })

        worksheet.set_column("A:A", 15)
        worksheet.set_column("B:B", 50)
        worksheet.set_column("C:C", 12)
        worksheet.set_column("D:D", 10)
        worksheet.set_column("E:E", 18)
        worksheet.set_column("F:F", 18)

        # Write header
        for col_num, col_name in enumerate(df_final.columns):
            worksheet.write(0, col_num, col_name, header_format)

        total_rows = len(df_final) + 1  # +1 for the header
        summary_rows = 3               # Three summary rows
        data_rows = total_rows - 1 - summary_rows

        # Write data rows
        for row in range(1, data_rows + 1):
            for col in range(6):
                cell_value = df_final.iloc[row - 1, col]
                if col == 1:
                    worksheet.write(row, col, cell_value, wrap_text_format)
                elif col in [4, 5]:
                    worksheet.write(row, col, cell_value, price_format)
                else:
                    worksheet.write(row, col, cell_value, data_format)

        # Write summary rows
        summary_start_row = data_rows + 1
        for i in range(summary_rows):
            current_row = summary_start_row + i
            label = df_final.iloc[data_rows + i, 1]
            value = df_final.iloc[data_rows + i, 5]
            worksheet.merge_range(current_row, 0, current_row, 4, label, summary_label_format)
            if label.strip() == "GRAND TOTAL":
                worksheet.write(current_row, 5, value, grand_total_format)
            else:
                worksheet.write(current_row, 5, value, summary_value_format)

        # Adjust row heights
        for row in range(total_rows):
            worksheet.set_row(row, 20)

    return file_path

def copy_sku_column(shipping_cost, boq_data):
    """
    Copies the SKU column from the final quote to the clipboard.
    Note: Clipboard functionality may not work on cloud platforms.
    """
    final_df = generate_final_quote(shipping_cost, boq_data)
    if final_df.empty:
        return "Final Quote is empty."
    sku_list = final_df["SKU"].dropna()
    sku_list = sku_list[sku_list != ""]
    sku_text = "\n".join(str(sku) for sku in sku_list)
    pyperclip.copy(sku_text)
    return "SKU column copied to clipboard!"

# ---------- Gradio UI Definition ----------
with gr.Blocks() as demo:
    gr.Markdown("""
<h1 style="text-align:center; margin-bottom:0;">
  🚀 Sophos Quotation Generator
</h1>
<p style="text-align:center; font-size:0.9em; margin-top:0;">
  Version 2 - Developed by Rajeesh Nair - rajeesh@starlinkme.net
</p>
""")

    # First row (General inputs)
    with gr.Row():
        category = gr.Dropdown(list(models_df.columns), label="Category")
        model_selected = gr.Dropdown([], label="Model")
        transaction_type = gr.Radio(["New", "Renewal"], label="Transaction Type")
        term = gr.Dropdown(["12 MOS", "24 MOS", "36 MOS"] + [f"{i} MOS" for i in range(1, 12)], label="Term")
        quantity = gr.Number(label="Quantity", value=1)
        manual_sku = gr.Textbox(label="Enter SKU (Optional)")
                
    with gr.Row():
        category_message = gr.Markdown("", visible=False)

    # Second row: Toggle and License dropdown for Firewall License
    with gr.Row():
        use_license = gr.Checkbox(label="Enable License", value=False, visible=False)
        license_box = gr.Dropdown(choices=[], label="License", visible=False)
    
    # Wireless License Toggle (no dropdown)
    with gr.Row():
        use_wireless_license = gr.Checkbox(label="Enable Wireless License", value=False, visible=False)
    
    # Switch Support License Toggle (only when category is "Switches")
    with gr.Row():
        use_switch_support = gr.Checkbox(label="Enable Switch Support License", value=False, visible=False)
    
    with gr.Row():
        override_shipping = gr.Checkbox(label="Override Default Shipping Cost", value=False)
        shipping_cost = gr.Number(label="Shipping Cost (%)", value=4, visible=False)
 
    # Fourth row: Partner Type, Deal Registration, and Incumbency
    with gr.Row():
        partner_type = gr.Dropdown(["Authorised", "Silver", "Gold", "Platinum"], label="Partner Type", value="Authorised")
        deal_registered = gr.Checkbox(label="Deal Registration", value=False, visible=True)
        incumbency = gr.Checkbox(label="Incumbency", value=False, visible=False)
    
    # Toggle deal registration/incumbency based on Transaction Type
    def toggle_registration(tx_type):
        if tx_type == "New":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    transaction_type.change(toggle_registration, inputs=[transaction_type], outputs=[deal_registered, incumbency])
    
    # State to store user's BOQ data
    quote_state = gr.State([])
    
    # Button actions
    with gr.Row():
        add_button = gr.Button("+ Add Line Item")
        recalc_button = gr.Button("Recalculate BOQ")
        remove_button = gr.Button("- Remove Last Item")
        reset_button = gr.Button("Reset Quote")
    
    gr.Markdown("## **BOQ**")
    boq_table = gr.Dataframe(
        headers=[
            "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
            "Reseller Price (USD)", "Disc. Price (USD)",
            "Override Discount (%)", "Additional Discount (%)",
            "Classification", "Product Type", "Status"
        ],
        interactive=True,
        label="BOQ Table (Editable: Quantity, Override Discount, Additional Discount)"
    )
    
    # Final Quote section
    gr.Markdown("<h2 style='color:green; font-weight:bold'>Final Quote</h2>")
    generate_button = gr.Button("Generate Final Quote")
    final_output = gr.Dataframe(label="Final Quote")
    
    download_button = gr.Button("Download Quote as Excel")
    download_output = gr.File(label="Download Excel File")
    
    copy_sku_btn = gr.Button("Copy SKU Column")
    copy_sku_output = gr.Textbox(label="Copy SKU Result")
    
    # ---------- Event bindings ----------
    category.change(update_models, inputs=[category], outputs=[model_selected])
    category.change(update_license_toggle, inputs=[category], outputs=[use_license])
    category.change(lambda cat, ul: update_license_box(ul, cat), inputs=[category, use_license], outputs=[license_box])
    use_license.change(update_license_box, inputs=[use_license, category], outputs=[license_box])
    category.change(update_category_message, inputs=[category], outputs=[category_message])
    override_shipping.change(toggle_shipping_box, inputs=[override_shipping], outputs=[shipping_cost])

    download_button.click(fn=download_final_quote, inputs=[boq_table], outputs=download_output)

    def toggle_wireless_license(model_sel):
        if model_sel.strip().upper().startswith("AP6"):
            return gr.update(visible=True)
        return gr.update(visible=False)

    model_selected.change(toggle_wireless_license, inputs=[model_selected], outputs=[use_wireless_license])
    
    def toggle_switch_support(category):
        if category.strip().lower() == "switches":
            return gr.update(visible=True, value=True)
        return gr.update(visible=False, value=True)

    category.change(toggle_switch_support, inputs=[category], outputs=[use_switch_support])
    
    add_button.click(
        add_line_item,
        inputs=[
            category, model_selected, transaction_type, term, quantity, partner_type, manual_sku,
            use_license, license_box, deal_registered, incumbency, use_wireless_license, use_switch_support, quote_state
        ],
        outputs=[boq_table, quote_state]
    )
    
    recalc_button.click(recalc_boQ, inputs=[boq_table], outputs=boq_table)
    remove_button.click(remove_last_item, inputs=[quote_state], outputs=[boq_table, quote_state])
    reset_button.click(reset_quote, inputs=[quote_state], outputs=[boq_table, quote_state])
    generate_button.click(generate_final_quote, inputs=[shipping_cost, boq_table], outputs=final_output)
    copy_sku_btn.click(copy_sku_column, inputs=[shipping_cost, boq_table], outputs=[copy_sku_output])
    
# Launch the Gradio app
demo.launch(server_name="0.0.0.0", server_port=7860)
