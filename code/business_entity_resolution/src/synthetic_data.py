"""Synthetic benchmark dataset generator for Amazon ML Challenge 2026.

Generates realistic sample datasets matching the exact schemas, noise patterns,
cardinality distributions, and open-set country requirements (US, India in train;
US, India, France in test).
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Set, Tuple
import pandas as pd


SAMPLE_BUSINESS_NAMES_US = [
    ("Apex Global Logistics Inc", "Apex Global Logistics", "Apex Global Log."),
    ("Horizon Health Systems LLC", "Horizon Health Systems", "Horizon Health"),
    ("Summit Financial Partners", "Summit Financial Partners Inc", "Summit Financial"),
    ("Pinnacle Retail Solutions", "Pinnacle Retail Sol", "Pinnacle Retail Solutions LLC"),
    ("Vanguard Technology Group", "Vanguard Tech Group", "Vanguard Technology"),
    ("Crestview Medical Center", "Crestview Medical Ctr", "Crestview Medical"),
    ("Metro Digital Innovations", "Metro Digital", "Metro Digital Innov."),
    ("Beacon Energy Services Corp", "Beacon Energy Services", "Beacon Energy Corp"),
    ("Starlight Hospitality LLC", "Starlight Hospitality", "Starlight Hotels"),
    ("Cascade Engineering Works", "Cascade Engg Works", "Cascade Engineering"),
]

SAMPLE_ADDRESSES_US = [
    ("100 Main Street, Suite 400, New York, NY 10001", "100 Main St, Ste 400, New York, 10001", "100 Main St NY NY"),
    ("450 Market Street, 5th Floor, San Francisco, CA 94105", "450 Market St, Fl 5, San Francisco 94105", "450 Market Street SF CA"),
    ("1200 Westheimer Road, Houston, TX 77006", "1200 Westheimer Rd, Houston, TX", "1200 Westheimer Road 77006"),
    ("800 Michigan Avenue, Chicago, IL 60611", "800 Michigan Ave, Chicago 60611", "800 N Michigan Ave Chicago IL"),
    ("3200 Wilshire Boulevard, Los Angeles, CA 90010", "3200 Wilshire Blvd, LA, CA 90010", "3200 Wilshire Blvd Los Angeles"),
]

SAMPLE_BUSINESS_NAMES_IN = [
    ("Reliance Retail Ventures Pvt Ltd", "Reliance Retail Ventures", "Reliance Retail Pvt Ltd"),
    ("Tata Consultancy Services Ltd", "Tata Consultancy Services", "TCS Ltd"),
    ("Infosys Digital Solutions Pvt Ltd", "Infosys Digital Solutions", "Infosys Digital"),
    ("HDFC Bank Financial Services", "HDFC Bank Financial", "HDFC Financial Services"),
    ("Bharti Airtel Telecom Ltd", "Bharti Airtel Telecom", "Airtel Telecom Ltd"),
    ("Mahindra Automotive Spares", "Mahindra Auto Spares Pvt Ltd", "Mahindra Automotive"),
    ("Larsen and Toubro Tech Services", "Larsen & Toubro Tech", "L&T Tech Services"),
    ("State Bank Merchant Services", "SBI Merchant Services", "State Bank Merchant"),
    ("Apollo Healthcare Clinics", "Apollo Healthcare Clinics Ltd", "Apollo Clinics"),
    ("Wipro Cloud Technologies", "Wipro Cloud Tech Pvt Ltd", "Wipro Cloud Tech"),
]

SAMPLE_ADDRESSES_IN = [
    ("Plot 42, Bandra Kurla Complex, Bandra East, Mumbai 400051", "Plot 42 BKC Near SBI ATM, Mumbai 400051", "Bandra Kurla Complex Mumbai 400051"),
    ("100 Feet Road, 12th Main, Indiranagar, Bangalore 560038", "100 Ft Rd Near Metro Station, Indiranagar, Bengaluru 560038", "100 Feet Rd Indiranagar Bangalore"),
    ("Sector 62, Electronic City, Phase 1, Noida 201309", "Sec 62 Electronic City, Noida 201309", "Sector 62 Noida UP"),
    ("Mount Road, Thousand Lights, Chennai 600006", "Mount Rd Opp Indian Bank, Chennai 600006", "Mount Road Chennai 600006"),
    ("Banjara Hills, Road No 12, Hyderabad 500034", "Rd No 12 Banjara Hills, Hyderabad 500034", "Banjara Hills Rd 12 Hyderabad"),
]

SAMPLE_BUSINESS_NAMES_FR = [
    ("Carrefour Supermarches SAS", "Carrefour Supermarches", "Carrefour SAS"),
    ("Societe Generale Banque SA", "Societe Generale", "SocGen Banque"),
    ("TotalEnergies Renouvelables SAS", "TotalEnergies Renouvelables", "Total Renouvelables"),
    ("Sanofi Sante Globale SARL", "Sanofi Sante Globale", "Sanofi SARL"),
    ("Danone Produits Alimentaires SAS", "Danone Produits Alimentaires", "Danone SAS"),
    ("Michelin Pneumatiques France", "Michelin Pneumatiques SA", "Michelin France"),
    ("L'Oreal Cosmetiques Paris SAS", "L'Oreal Cosmetiques", "L'Oreal Paris SAS"),
    ("Bouygues Construction SA", "Bouygues Construction", "Bouygues BTP"),
]

SAMPLE_ADDRESSES_FR = [
    ("15 Rue de Rivoli, 75004 Paris", "15 R. de Rivoli, Paris 75004", "15 Rue de Rivoli 75004"),
    ("42 Avenue des Champs-Elysees, 75008 Paris", "42 Av. Champs Elysees, Paris 75008", "42 Avenue Champs Elysees"),
    ("10 Rue de la Republique, 69002 Lyon", "10 R de la Republique, Lyon 69002", "10 Rue Republique Lyon"),
    ("25 Boulevard Michelet, 13008 Marseille", "25 Blvd Michelet, Marseille 13008", "25 Boulevard Michelet Marseille"),
    ("8 Place du Capitole, 31000 Toulouse", "8 Pl du Capitole, Toulouse 31000", "8 Place Capitole Toulouse"),
]


def generate_benchmark_dataset(
    output_dir: str,
    n_train_s1: int = 100,
    n_test_s1: int = 50,
    random_state: int = 42,
) -> None:
    """Generate sample train and test datasets matching the exact competition specs.

    Creates:
    - dataset/train/train_source1.tsv
    - dataset/train/train_source2.tsv
    - dataset/train/train_source3.tsv
    - dataset/train/train_ground_truth.tsv
    - dataset/test/test_source1.tsv
    - dataset/test/test_source2.tsv
    - dataset/test/test_source3.tsv
    """
    random.seed(random_state)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def build_dataset_records(
        n_entities: int,
        countries: List[str],
        s1_start_id: int,
        s2_start_id: int,
        s3_start_id: int,
        include_ground_truth: bool,
    ) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
        s1_records, s2_records, s3_records, gt_records = [], [], [], []
        s2_id_counter = s2_start_id
        s3_id_counter = s3_start_id

        for i in range(n_entities):
            s1_id = f"S1-{s1_start_id + i:05d}"
            country = random.choice(countries)

            if country == "US":
                names, addrs = SAMPLE_BUSINESS_NAMES_US, SAMPLE_ADDRESSES_US
            elif country == "India":
                names, addrs = SAMPLE_BUSINESS_NAMES_IN, SAMPLE_ADDRESSES_IN
            else:
                names, addrs = SAMPLE_BUSINESS_NAMES_FR, SAMPLE_ADDRESSES_FR

            name_tpl = random.choice(names)
            addr_tpl = random.choice(addrs)

            s1_records.append({
                "entity_id": s1_id,
                "business_name": name_tpl[0],
                "business_address": addr_tpl[0],
                "country": country,
            })

            # Match cardinality: 30% singletons, 50% 1-to-1, 20% 1-to-many
            rand_val = random.random()
            matched_ids = []

            if rand_val < 0.30:
                # Singleton: 0 matches
                pass
            elif rand_val < 0.80:
                # 1-to-1 match: match in S2 or S3
                target_src = random.choice([2, 3])
                if target_src == 2:
                    mid = f"S2-{s2_id_counter:05d}"
                    s2_id_counter += 1
                    s2_records.append({
                        "entity_id": mid,
                        "business_name": name_tpl[1],
                        "business_address": addr_tpl[1],
                        "country": country,
                    })
                else:
                    mid = f"S3-{s3_id_counter:05d}"
                    s3_id_counter += 1
                    s3_records.append({
                        "entity_id": mid,
                        "business_name": name_tpl[2],
                        "business_address": addr_tpl[2],
                        "country": country,
                    })
                matched_ids.append(mid)
            else:
                # 1-to-many: match in both S2 and S3 (or multiple in one)
                mid_s2 = f"S2-{s2_id_counter:05d}"
                s2_id_counter += 1
                s2_records.append({
                    "entity_id": mid_s2,
                    "business_name": name_tpl[1],
                    "business_address": addr_tpl[1],
                    "country": country,
                })
                matched_ids.append(mid_s2)

                mid_s3 = f"S3-{s3_id_counter:05d}"
                s3_id_counter += 1
                s3_records.append({
                    "entity_id": mid_s3,
                    "business_name": name_tpl[2],
                    "business_address": addr_tpl[2],
                    "country": country,
                })
                matched_ids.append(mid_s3)

            if include_ground_truth:
                gt_records.append({
                    "source1_entity_id": s1_id,
                    "matched_entity_ids": ",".join(matched_ids),
                })

        # Add distractor entities in S2 and S3 (records that do NOT match any S1)
        n_distractors = int(n_entities * 0.25)
        for _ in range(n_distractors):
            country = random.choice(countries)
            if country == "US":
                names, addrs = SAMPLE_BUSINESS_NAMES_US, SAMPLE_ADDRESSES_US
            elif country == "India":
                names, addrs = SAMPLE_BUSINESS_NAMES_IN, SAMPLE_ADDRESSES_IN
            else:
                names, addrs = SAMPLE_BUSINESS_NAMES_FR, SAMPLE_ADDRESSES_FR

            name_tpl = random.choice(names)
            addr_tpl = random.choice(addrs)

            s2_mid = f"S2-{s2_id_counter:05d}"
            s2_id_counter += 1
            s2_records.append({
                "entity_id": s2_mid,
                "business_name": f"Distractor {name_tpl[0]}",
                "business_address": f"Unrelated {addr_tpl[0]}",
                "country": country,
            })

            s3_mid = f"S3-{s3_id_counter:05d}"
            s3_id_counter += 1
            s3_records.append({
                "entity_id": s3_mid,
                "business_name": f"Other {name_tpl[1]}",
                "business_address": f"Different {addr_tpl[1]}",
                "country": country,
            })

        # Shuffle target records to avoid positional bias
        random.shuffle(s2_records)
        random.shuffle(s3_records)

        return s1_records, s2_records, s3_records, gt_records

    # 1. Training set: US and India
    train_s1, train_s2, train_s3, train_gt = build_dataset_records(
        n_entities=n_train_s1,
        countries=["US", "India"],
        s1_start_id=1,
        s2_start_id=1,
        s3_start_id=1,
        include_ground_truth=True,
    )

    pd.DataFrame(train_s1).to_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", index=False)
    pd.DataFrame(train_s2).to_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", index=False)
    pd.DataFrame(train_s3).to_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", index=False)
    pd.DataFrame(train_gt).to_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    # 2. Test set: US, India, and France (open-set country)
    test_s1, test_s2, test_s3, _ = build_dataset_records(
        n_entities=n_test_s1,
        countries=["US", "India", "France"],
        s1_start_id=n_train_s1 + 1,
        s2_start_id=len(train_s2) + 1,
        s3_start_id=len(train_s3) + 1,
        include_ground_truth=False,
    )

    pd.DataFrame(test_s1).to_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", index=False)
    pd.DataFrame(test_s2).to_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", index=False)
    pd.DataFrame(test_s3).to_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", index=False)

    print(f"Generated sample datasets successfully in {output_dir}:")
    print(f"  Train: S1={len(train_s1)}, S2={len(train_s2)}, S3={len(train_s3)}, GT={len(train_gt)}")
    print(f"  Test:  S1={len(test_s1)}, S2={len(test_s2)}, S3={len(test_s3)} (including France)")
