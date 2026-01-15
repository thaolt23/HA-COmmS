def print_ha_report(report):
    print("=== HA Explainability Report ===")
    for k, v in report.items():
        print(f"{k:12s}: {v:5.2f}%")
