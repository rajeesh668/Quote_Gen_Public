import gradio as gr
import pandas as pd
import re
import os
import pyperclip  # Ensure you have pyperclip installed
import uuid
import time

# ---------------------
# Data Loading Functions
# ---------------------
def load_data():
    """
    Loads Models.csv and SKU.csv from the local folder.
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
    Loads License.csv from the local folder.
    """
    try:
        llicense_df = pd.read_csv("data/License.csv", encoding="ISO-8859-1")
        return license_df
    except Exception as e:
        print("Error loading License CSV:", e)
        return pd.DataFrame()

def load_fixed_skus():
    """
    Loads FixedSKUs.csv from the local folder.
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
# Currency Conversion and VAT Setup
# ---------------------
conversion_rates = {
    "SAR": 3.7575,  # Default value; for SAR
    "AED": 3.67,
    "QAR": 3.722,
    "OMR": 0.3846,
    "KWD": 0.32,
    "USD": 1.0,
}

# ---------------------
# Helper: Custom Rounding Function
# ---------------------
def custom_round(x):
    """
    Round x to the nearest integer:
    - If fractional part is 0.5 or more, round up.
    - Otherwise, round down.
    """
    return int(x + 0.5)

# ---------------------
# Helper Functions using gr.State instead of a global variable
# ---------------------
def remove_last_item(quote_state):
    """
    Removes the last item from the state.
    """
    if isinstance(quote_state, pd.DataFrame):
        state_list = quote_state.values.tolist()
    else:
        state_list = quote_state if quote_state is not None else []
    if len(state_list) > 0:
        state_list.pop()
    df = pd.DataFrame(state_list, columns=[
        "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
        "Reseller Price (USD)", "Disc. Price (USD)",
        "Override Discount (%)", "Additional Discount (%)",
        "Classification", "Product Type", "Status"
    ])
    return df, pd.DataFrame(state_list, columns=df.columns)

def reset_quote(quote_state):
    """
    Resets the state to an empty list.
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
    if not models_df.empty and category in models_df.columns:
        models_list = models_df[category].dropna().tolist()
        return gr.update(choices=models_list, value=models_list[0] if models_list else "")
    return gr.update(choices=[], value="")

def update_license_toggle(category):
    if str(category).lower() == "firewall":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False, value=False)

def update_license_box(use_license, category):
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

def update_category_message(selected_category):
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

def update_switch_support_toggle(category):
    if str(category).strip().lower() == "switches":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False, value=False)

def find_matching_sku(category, model_selected, transaction_type, term, quantity, manual_sku="", license_selected=""):
    sku_input = manual_sku.strip().lower()
    if sku_input:
        sku_details = sku_df[sku_df["SKU"].str.lower() == sku_input]
        if not sku_details.empty:
            row = sku_details.iloc[0]
            return row["SKU"], row["Description"], float(row["Price"]), row["Product Type"], term
        else:
            return manual_sku.strip(), "Manual SKU Not Found", 0.0, "Unknown", term
    if not (str(category).lower() == "firewall" and license_selected):
        if model_selected in fixed_skus:
            sku = fixed_skus[model_selected]
            sku_details = sku_df[sku_df["SKU"] == sku]
            if not sku_details.empty:
                description = sku_details.iloc[0]["Description"]
                price = float(sku_details.iloc[0]["Price"])
                product_type = sku_details.iloc[0]["Product Type"]
                return sku, description, price, product_type, "-"
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
    if str(category).lower() == "firewall" and license_selected:
        filtered_sku = filtered_sku[~filtered_sku["Product Type"].isin(["Hardware", "Appliance"])]
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
    if not filtered_sku.empty:
        row = filtered_sku.iloc[0]
        return row["SKU"], row["Description"], float(row["Price"]), row["Product Type"], term
    return "", "No matching SKU found", 0.0, "", term

def find_wireless_license_support_sku(model_name, transaction_type, term):
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

def find_switch_support_sku(model_name, transaction_type, term):
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

def add_line_item(category, model_selected, transaction_type, term, quantity, partner_type, manual_sku,
                  use_license, license_selected, deal_registered, incumbency, use_wireless_license,
                  use_switch_support, current_boq):
    """
    Adds new line items without resetting already edited BOQ values.
    Uses the current BOQ table (passed as current_boq) to preserve manual edits.
    """
    if current_boq is not None and not current_boq.empty:
        state = current_boq.values.tolist()
    else:
        state = []
    
    # ----- Wireless License Branch -----
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
            return df, pd.DataFrame(state, columns=df.columns)
    
    # ----- Switch Support Branch -----
    if use_switch_support and str(category).strip().lower() == "switches":
        sw_sku, sw_desc, sw_price, sw_type = find_switch_support_sku(
            model_selected, transaction_type, term
        )
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
            return df, pd.DataFrame(state, columns=df.columns)
    
    sku_list = [line.strip() for line in manual_sku.splitlines() if line.strip()] if manual_sku.strip() else [manual_sku]
    
    if sku_list and len(sku_list) > 1:
        for sku_input in sku_list:
            sku_val, desc, orig_price, prod_type, adj_term = find_matching_sku(
                category, model_selected, transaction_type, term, quantity, sku_input, license_selected
            )
            if prod_type.strip().lower() in ["hardware", "appliance", "proservices"]:
                adj_term = "-"
            sku_rows = sku_df[sku_df["SKU"].str.lower() == sku_val.strip().lower()]
            if not sku_rows.empty:
                raw_class = sku_rows.iloc[0].get("Classification", "")
                prod_class = str(raw_class).strip().upper() if raw_class and raw_class.strip() != "" else prod_type.upper()
            else:
                prod_class = prod_type.upper()
            trans_type = sku_rows.iloc[0].get("Transaction Type", transaction_type) if not sku_rows.empty else transaction_type
            if prod_class in ["SMB", "WIRELESS LICENSE", "SWITCH SUPPORT"]:
                if orig_price * quantity >= 5000:
                    prod_class = "MME"
                    if trans_type == "New":
                        if deal_registered:
                            disc_rate = partner_discounts_DR.get(partner_type, 0)
                            status_txt = f"MME, DR: {disc_rate*100:.0f}%"
                        else:
                            disc_rate = 0.10
                            status_txt = "MME, No DR: 10%"
                    elif trans_type == "Renewal":
                        if incumbency:
                            disc_rate = partner_incumbency.get(partner_type, 0)
                            status_txt = f"MME, Incumbency: {disc_rate*100:.0f}%"
                        else:
                            disc_rate = 0.10
                            status_txt = "MME, No Incumbency: 10%"
                    else:
                        disc_rate = 0.10
                        status_txt = "MME, 10%"
                else:
                    if trans_type == "New":
                        if prod_class == "MME":
                            if deal_registered:
                                disc_rate = partner_discounts_DR.get(partner_type, 0)
                                status_txt = f"{prod_class}, DR: {disc_rate*100:.0f}%"
                            else:
                                disc_rate = 0.10
                                status_txt = f"{prod_class}, No DR: 10%"
                        else:
                            disc_rate = partner_discounts.get(partner_type, 0)
                            status_txt = f"{prod_class}, {disc_rate*100:.0f}%"
                    elif trans_type == "Renewal":
                        if prod_class == "MME":
                            if incumbency:
                                disc_rate = partner_incumbency.get(partner_type, 0)
                                status_txt = f"{prod_class}, Incumbency: {disc_rate*100:.0f}%"
                            else:
                                disc_rate = 0.10
                                status_txt = f"{prod_class}, No Incumbency: 10%"
                        else:
                            disc_rate = partner_discounts.get(partner_type, 0)
                            status_txt = f"{prod_class}, {disc_rate*100:.0f}%"
                    else:
                        disc_rate = 0.10
                        status_txt = f"{prod_class}, 10%"
            else:
                prod_class = "Non-Core"
                disc_rate = 0.10
                status_txt = "Non-Core, 10%"
            res_price = orig_price * (1 - disc_rate)
            disc_price = res_price
            state.append([
                sku_val, desc, adj_term, quantity, f"{orig_price:.2f}",
                f"{res_price:.2f}", f"{disc_price:.2f}",
                f"{disc_rate*100:.0f}", "0",
                prod_class, prod_type, status_txt
            ])
        df = pd.DataFrame(state, columns=[
            "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
            "Reseller Price (USD)", "Disc. Price (USD)",
            "Override Discount (%)", "Additional Discount (%)",
            "Classification", "Product Type", "Status"
        ])
        return df, pd.DataFrame(state, columns=df.columns)
    else:
        if not use_license:
            license_selected = ""
        sku, description, orig_price, prod_type, adj_term = find_matching_sku(
            category, model_selected, transaction_type, term, quantity, manual_sku, license_selected
        )
        if prod_type.strip().lower() in ["hardware", "appliance", "proservices"]:
            adj_term = "-"
        if not sku:
            return pd.DataFrame(state, columns=[
                "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
                "Reseller Price (USD)", "Disc. Price (USD)",
                "Override Discount (%)", "Additional Discount (%)",
                "Classification", "Product Type", "Status"
            ]), pd.DataFrame(state, columns=[
                "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
                "Reseller Price (USD)", "Disc. Price (USD)",
                "Override Discount (%)", "Additional Discount (%)",
                "Classification", "Product Type", "Status"
            ])
        sku_rows = sku_df[sku_df["SKU"].str.lower() == sku.strip().lower()]
        trans_type = sku_rows.iloc[0].get("Transaction Type", transaction_type) if not sku_rows.empty else transaction_type
        discount_rate = 0.10
        classification = "-"
        status_text = ""
        if not sku_rows.empty:
            val = sku_rows.iloc[0].get("Classification", "")
            raw_class = str(val).strip().upper()
            if raw_class in ["SMB", "MME"]:
                classification = raw_class
            else:
                classification = "-"
        if classification in ["SMB", "MME"]:
            if trans_type == "New":
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
            elif trans_type == "Renewal":
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
            status_text = "Non-Core, 10%"
        if classification.upper() in ["SMB", "WIRELESS LICENSE", "SWITCH SUPPORT"]:
            if orig_price * quantity >= 5000:
                classification = "MME"
                if trans_type == "New":
                    if deal_registered:
                        discount_rate = partner_discounts_DR.get(partner_type, 0)
                        status_text = "MME, DR: " + f"{discount_rate*100:.0f}%"
                    else:
                        discount_rate = 0.10
                        status_text = "MME, No DR: 10%"
                elif trans_type == "Renewal":
                    if incumbency:
                        discount_rate = partner_incumbency.get(partner_type, 0)
                        status_text = "MME, Incumbency: " + f"{discount_rate*100:.0f}%"
                    else:
                        discount_rate = 0.10
                        status_text = "MME, No Incumbency: 10%"
                else:
                    discount_rate = 0.10
                    status_text = "MME, 10%"
        res_price = orig_price * (1 - discount_rate)
        disc_price = res_price
        state.append([
            sku, description, adj_term, quantity, f"{orig_price:.2f}",
            f"{res_price:.2f}", f"{disc_price:.2f}",
            f"{discount_rate*100:.0f}", "0",
            classification, prod_type, status_text
        ])
        df = pd.DataFrame(state, columns=[
            "SKU", "Description", "Term", "Quantity", "Original Price (USD)",
            "Reseller Price (USD)", "Disc. Price (USD)",
            "Override Discount (%)", "Additional Discount (%)",
            "Classification", "Product Type", "Status"
        ])
        return df, pd.DataFrame(state, columns=df.columns)

def recalc_boQ(boq_df, partner_type, ui_trans_type, deal_registered, incumbency):
    """
    Recalculates the BOQ and, if the cumulative Original Price*Qty of core products is >=5000,
    then for any line item that is classified as "SMB" and qualifies based on UI selections (New with Deal Registration
    or Renewal with Incumbency), automatically override the Classification to "MME" and set the Override Discount (%) accordingly.
    """
    # First, compute the cumulative total for core products (we consider "SMB" and "MME" as core)
    cumulative_total = 0.0
    for idx, row in boq_df.iterrows():
        classification = str(row.get("Classification", "")).strip().upper()
        if classification in ["SMB", "MME"]:
            try:
                original = float(row["Original Price (USD)"])
                qty = float(row["Quantity"])
            except:
                continue
            cumulative_total += original * qty

    # If cumulative total is 5000 or more, adjust qualifying lines
    if cumulative_total >= 5000:
        for idx, row in boq_df.iterrows():
            classification = str(row.get("Classification", "")).strip().upper()
            # Check if line is a core product that is still "SMB"
            if classification == "SMB":
                # Use the UI transaction type (ui_trans_type) to decide which condition to apply
                if ui_trans_type == "New" and deal_registered:
                    # Override discount using partner_discounts_DR
                    new_disc = partner_discounts_DR.get(partner_type, 0) * 100  # percentage
                    boq_df.at[idx, "Classification"] = "MME"
                    boq_df.at[idx, "Override Discount (%)"] = f"{new_disc:.0f}"
                    boq_df.at[idx, "Status"] = f"MME, DR: {new_disc:.0f}%"
                elif ui_trans_type == "Renewal" and incumbency:
                    new_disc = partner_incumbency.get(partner_type, 0) * 100
                    boq_df.at[idx, "Classification"] = "MME"
                    boq_df.at[idx, "Override Discount (%)"] = f"{new_disc:.0f}"
                    boq_df.at[idx, "Status"] = f"MME, Incumbency: {new_disc:.0f}%"
    # Now, proceed with final price recalculation as before
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
        final_price = custom_round(final_price)
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

def generate_final_quote_wrapper(shipping_cost, boq_table, currency, vat, partner_type, ui_trans_type, deal_registered, incumbency):
    time.sleep(0.5)  # slight delay to allow manual edits to commit
    updated_boq = recalc_boQ(boq_table, partner_type, ui_trans_type, deal_registered, incumbency)
    final_quote = generate_final_quote(shipping_cost, updated_boq, currency, vat)
    return final_quote, updated_boq

def generate_final_quote(shipping_cost, boq_data, currency, vat):
    if shipping_cost is None or shipping_cost == "":
        shipping_cost = 4
    if boq_data is None or boq_data.empty:
        return pd.DataFrame(columns=[
            "SKU", "Description", "Term", "Quantity", 
            f"Unit Price ({currency})", f"Total Price ({currency})"
        ])
    df_quote = boq_data.copy()
    conversion_rate = conversion_rates.get(currency, 1.0)
    df_quote[f"Unit Price ({currency})"] = df_quote["Disc. Price (USD)"].astype(float).apply(lambda x: custom_round(x * conversion_rate))
    df_quote[f"Total Price ({currency})"] = df_quote.apply(lambda row: custom_round(row[f"Unit Price ({currency})"] * float(row["Quantity"])), axis=1)
    total_subtotal = df_quote[f"Total Price ({currency})"].sum()
    shipping_items = df_quote[df_quote["Product Type"].isin(["Appliance", "Hardware"])]
    total_original_usd = shipping_items.apply(lambda r: float(r["Original Price (USD)"]) * float(r["Quantity"]), axis=1).sum()
    total_shipping = total_original_usd * (float(shipping_cost) / 100.0) * conversion_rate
    total_vat = (vat/100.0) * (total_subtotal + total_shipping)
    grand_total = total_subtotal + total_shipping + total_vat
    df_final_quote = df_quote[[
        "SKU", "Description", "Term", "Quantity", 
        f"Unit Price ({currency})", f"Total Price ({currency})"
    ]]
    df_summary = pd.DataFrame([
        ["", "Shipping Cost", "", "", "", f"{total_shipping:.2f}"],
        ["", f"VAT ({vat}%)", "", "", "", f"{total_vat:.2f}"],
        ["", "GRAND TOTAL", "", "", "", f"{grand_total:.2f}"]
    ], columns=df_final_quote.columns)
    df_final = pd.concat([df_final_quote, df_summary], ignore_index=True)
    return df_final

def toggle_shipping_box(override_value):
    if override_value:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False, value=4)

def download_final_quote(boq_data, currency, vat):
    if boq_data is None or boq_data.empty:
        return None
    df_final = generate_final_quote(4, boq_data, currency, vat)
    file_path = os.path.join(f"Final_Quote_{uuid.uuid4().hex}.xlsx")
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

        for col_num, col_name in enumerate(df_final.columns):
            worksheet.write(0, col_num, col_name, header_format)

        total_rows = len(df_final) + 1  # +1 for header
        summary_rows = 3               # Three summary rows
        data_rows = total_rows - 1 - summary_rows

        for row in range(1, data_rows + 1):
            for col in range(6):
                cell_value = df_final.iloc[row - 1, col]
                if col == 1:
                    worksheet.write(row, col, cell_value, wrap_text_format)
                elif col in [4, 5]:
                    worksheet.write(row, col, cell_value, price_format)
                else:
                    worksheet.write(row, col, cell_value, data_format)

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

        for row in range(total_rows):
            worksheet.set_row(row, 20)

    return file_path

def copy_sku_column(shipping_cost, boq_data):
    final_df = generate_final_quote(shipping_cost, boq_data, "SAR", 15)
    if final_df.empty:
        return "Final Quote is empty."
    sku_list = final_df["SKU"].dropna()
    sku_list = sku_list[sku_list != ""]
    sku_text = "\n".join(str(sku) for sku in sku_list)
    pyperclip.copy(sku_text)
    return "SKU column copied to clipboard!"

def toggle_shipping_box(override_value):
    if override_value:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False, value=4)

# ---------------------
# Gradio UI Definition
# ---------------------
with gr.Blocks() as demo:
    gr.Markdown("""
<h1 style="text-align:center; margin-bottom:0;">
  🚀 Sophos Quotation Generator
</h1>
<p style="text-align:center; font-size:0.9em; margin-top:0;">
  Version 2 - Developed by Rajeesh Nair - rajeesh@starlinkme.net
</p>
    """)
    
    with gr.Row():
        category = gr.Dropdown(list(models_df.columns), label="Category")
        model_selected = gr.Dropdown([], label="Model")
        transaction_type = gr.Radio(["New", "Renewal"], label="Transaction Type")
        term = gr.Dropdown(["12 MOS", "24 MOS", "36 MOS"] + [f"{i} MOS" for i in range(1, 12)], label="Term")
        quantity = gr.Number(label="Quantity", value=1)
        manual_sku = gr.Textbox(label="Enter SKU (Optional)", lines=5)
                
    with gr.Row():
        category_message = gr.Markdown("", visible=False)
    with gr.Row():
        use_license = gr.Checkbox(label="Enable License", value=False, visible=False)
        license_box = gr.Dropdown(choices=[], label="License", visible=False)
    with gr.Row():
        use_wireless_license = gr.Checkbox(label="Enable Wireless License", value=False, visible=False)
    with gr.Row():
        use_switch_support = gr.Checkbox(label="Enable Switch Support License", value=False, visible=False)
    with gr.Row():
        override_shipping = gr.Checkbox(label="Override Default Shipping Cost", value=False)
        shipping_cost = gr.Number(label="Shipping Cost (%)", value=4, visible=False)
    with gr.Row():
        currency = gr.Dropdown(choices=["SAR", "AED", "QAR", "OMR", "KWD", "USD"], label="Currency", value="SAR")
        vat = gr.Number(label="VAT (%)", value=15)
    with gr.Row():
        partner_type = gr.Dropdown(["Authorised", "Silver", "Gold", "Platinum"], label="Partner Type", value="Authorised")
        deal_registered = gr.Checkbox(label="Deal Registration", value=False, visible=True)
        incumbency = gr.Checkbox(label="Incumbency", value=False, visible=True)
    
    # Use hidden state (gr.State) to store the BOQ data.
    quote_state = gr.State([])
    
    with gr.Row():
        add_button = gr.Button("+ Add Line Item")
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
    download_button.click(fn=download_final_quote, inputs=[boq_table, currency, vat], outputs=[download_output])
    
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
            use_license, license_box, deal_registered, incumbency, use_wireless_license, use_switch_support, boq_table
        ],
        outputs=[boq_table, quote_state]
    )
    
    remove_button.click(remove_last_item, inputs=[quote_state], outputs=[boq_table, quote_state])
    reset_button.click(reset_quote, inputs=[quote_state], outputs=[boq_table, quote_state])
    
    # Updated generate final quote wrapper now accepts extra UI parameters for core total check
    generate_button.click(
        generate_final_quote_wrapper,
        inputs=[shipping_cost, boq_table, currency, vat, partner_type, transaction_type, deal_registered, incumbency],
        outputs=[final_output, boq_table]
    )
    
    copy_sku_btn.click(copy_sku_column, inputs=[shipping_cost, boq_table], outputs=copy_sku_output)
    
# Launch the Gradio app with port from environment variable.
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
