import pandas as pd
from shapely.geometry import Polygon

# SIMULASI DATA GEOMETRI SPASIAL BIDANG LAHAN
parcels_data = [
    {
        "parcel_id": "P001",
        "owner": "Kelompok Tani Makmur",
        "initial_type": "Pertanian Padi",
        "coordinates": [(102.25, -3.80), (102.26, -3.80), (102.26, -3.81), (102.25, -3.81)]
    },
    {
        "parcel_id": "P002",
        "owner": "PT Perumahan Asri",
        "initial_type": "Pertanian Padi",
        "coordinates": [(102.27, -3.82), (102.28, -3.82), (102.28, -3.83), (102.27, -3.83)]
    },
    {
        "parcel_id": "P003",
        "owner": "H. Sukri",
        "initial_type": "Pertanian Padi",
        "coordinates": [(102.29, -3.84), (102.295, -3.84), (102.295, -3.845), (102.29, -3.845)]
    }
]

# DATA SURVEI LAPANGAN TERBARU
field_survey = {
    "P001": {"current_status": "Padi Produktif", "building_density_pct": 2.0},
    "P002": {"current_status": "Konstruksi Perumahan", "building_density_pct": 78.5},
    "P003": {"current_status": "Padi Produktif", "building_density_pct": 0.0}
}

def calculate_area_ha(coords):
    poly = Polygon(coords)
    area_m2 = poly.area * (111320 ** 2)
    return round(area_m2 / 10000, 2)

def process_gis_landuse(parcels, survey):
    records = []
    for p in parcels:
        pid = p["parcel_id"]
        area = calculate_area_ha(p["coordinates"])
        surv = survey.get(pid, {"current_status": "Unknown", "building_density_pct": 0.0})
        
        status = "INDIKASI PELANGGARAN LP2B" if p["initial_type"] == "Pertanian Padi" and surv["building_density_pct"] > 20.0 else "AMAN LP2B"
        
        records.append({
            "Parcel_ID": pid,
            "Pemilik": p["owner"],
            "Status_Lapangan": surv["current_status"],
            "Luas_Ha": area,
            "Kepadatan_Bangunan_%": surv["building_density_pct"],
            "Kategori_Analisis": status
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    print("=== TOPIK 1: OTOMASI ANALISIS SPASIAL PENGGUNAAN LAHAN (GIS) ===")
    df = process_gis_landuse(parcels_data, field_survey)
    print(df.to_string(index=False))