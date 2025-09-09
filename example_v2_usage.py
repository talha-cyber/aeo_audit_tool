#!/usr/bin/env python3
"""
Example usage of Report Generator v2.

This script demonstrates how to generate v2 reports with the new engine.
"""

from app.db.session import SessionLocal
from app.services.report_generator import ReportGenerator


def generate_v2_report_example():
    """Example of generating a v2 report."""

    # Get database session
    db = SessionLocal()

    try:
        # Initialize report generator
        generator = ReportGenerator(db)

        # Example audit run ID (replace with actual ID from your database)
        audit_run_id = "your-audit-run-id-here"

        print("Generating v2 comprehensive report...")

        # Generate v2 report
        report_path = generator.generate_audit_report(
            audit_run_id=audit_run_id,
            report_type="v2_comprehensive",  # Uses default theme
            output_dir="reports/",
        )

        print(f"✅ V2 report generated successfully: {report_path}")

        # Generate enhanced corporate theme report
        print("Generating v2 enhanced report...")

        enhanced_report_path = generator.generate_audit_report(
            audit_run_id=audit_run_id,
            report_type="v2_enhanced",  # Uses corporate theme
            output_dir="reports/",
        )

        print(f"✅ V2 enhanced report generated successfully: {enhanced_report_path}")

        return [report_path, enhanced_report_path]

    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a valid audit_run_id with completed data.")
        return None

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

    finally:
        db.close()


def validate_audit_data_example():
    """Example of validating audit data before report generation."""

    db = SessionLocal()

    try:
        from app.reports.v2.engine import ReportEngineV2

        # Initialize v2 engine
        engine = ReportEngineV2(db, theme_key="default")

        # Example audit run ID
        audit_run_id = "your-audit-run-id-here"

        print("Validating audit data...")

        # Validate data
        validation_result = engine.validate_audit_data(audit_run_id)

        print(
            f"Validation result: {'✅ Valid' if validation_result['valid'] else '❌ Issues found'}"
        )

        if validation_result["issues"]:
            print("Issues:")
            for issue in validation_result["issues"]:
                print(f"  - {issue}")

        if validation_result["recommendations"]:
            print("Recommendations:")
            for rec in validation_result["recommendations"]:
                print(f"  - {rec}")

        if "data_summary" in validation_result:
            summary = validation_result["data_summary"]
            print("Data Summary:")
            print(f"  - Total responses: {summary.get('total_responses', 0)}")
            print(f"  - Platforms: {', '.join(summary.get('platforms', []))}")
            print(f"  - Brands analyzed: {summary.get('brands_analyzed', 0)}")

        return validation_result

    except Exception as e:
        print(f"❌ Validation error: {e}")
        return None

    finally:
        db.close()


def list_available_themes():
    """List all available themes for v2 reports."""

    db = SessionLocal()

    try:
        from app.reports.v2.engine import ReportEngineV2

        engine = ReportEngineV2(db)
        themes = engine.get_available_themes()

        print("Available v2 themes:")
        for theme in themes:
            print(f"  - {theme}")

        return themes

    except Exception as e:
        print(f"❌ Error listing themes: {e}")
        return []

    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Report Generator v2 Usage Examples")
    print("=" * 60)

    # List available themes
    print("\n1. Available Themes:")
    list_available_themes()

    # Validate audit data
    print("\n2. Data Validation Example:")
    validation_result = validate_audit_data_example()

    # Generate reports (commented out - requires valid audit_run_id)
    print("\n3. Report Generation Example:")
    print("   (Update audit_run_id in the script to test)")
    # report_paths = generate_v2_report_example()

    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("1. Run database migration first: alembic upgrade head")
    print("2. Replace 'your-audit-run-id-here' with actual audit run ID")
    print("3. Ensure audit run has completed successfully with data")
    print("4. Check reports/ directory for generated PDF files")
    print("=" * 60)
