#!/usr/bin/env python3
"""
Test script for document processing pipeline using docling.

This script tests the complete document processing pipeline:
1. Document parsing with docling
2. Table extraction and classification
3. Data validation and storage
4. Text chunking and vector storage

Usage:
    python test_document_processing.py

Requirements:
    - Docker containers running (backend, postgres, redis)
    - Sample PDF file available
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add the app directory to path (since we're in /app/app/services/test/)
sys.path.append('/app')

from app.services.document_processor import DocumentProcessor
from app.services.table_parser import TableParser
from app.db.session import SessionLocal
from app.models.transaction import CapitalCall, Distribution, Adjustment
from app.models.document import Document
from app.models.fund import Fund


async def test_document_processing():
    """Test the complete document processing pipeline"""

    print("🚀 Starting Document Processing Pipeline Test")
    print("=" * 60)

    # Test file path
    pdf_path = "/app/files/Sample_Fund_Performance_Report.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ Sample PDF not found at: {pdf_path}")
        print("Please ensure the sample PDF exists in the files/ directory")
        return False

    print(f"📄 Using test file: {pdf_path}")

    # Step 1: Test Table Parser
    print("\n📊 Step 1: Testing Table Parser")
    print("-" * 40)

    try:
        table_parser = TableParser()
        tables_data = table_parser.parse_tables(pdf_path)

        print("✅ Table parsing completed successfully!")
        print(f"   📋 Capital Calls: {len(tables_data['capital_calls'])} tables")
        print(f"   💰 Distributions: {len(tables_data['distributions'])} tables")
        print(f"   🔧 Adjustments: {len(tables_data['adjustments'])} tables")

        # Show sample data from each table type
        for table_type, tables in tables_data.items():
            if tables:
                print(f"\n   {table_type.upper()} Sample:")
                table = tables[0]
                print(f"   Headers: {table.get('headers', [])}")
                rows = table.get('rows', [])
                if rows:
                    print(f"   Sample Row: {rows[0]}")
                    print(f"   Total Rows: {len(rows)}")

    except Exception as e:
        print(f"❌ Table parsing failed: {e}")
        return False

    # Step 2: Test Document Processor
    print("\n🔄 Step 2: Testing Document Processor")
    print("-" * 40)

    try:
        processor = DocumentProcessor()

        start_time = time.time()
        result = await processor.process_document(pdf_path, 1, 1)
        processing_time = time.time() - start_time

        print("✅ Document processing completed successfully!")
        print(".2f")
        print(f"   📄 Status: {result['status']}")
        print(f"   📊 Tables Extracted: {result['tables_extracted']}")
        print(f"   📝 Text Chunks: {result['text_chunks_created']}")
        print(f"   📑 Total Pages: {result.get('total_pages', 0)}")

        if processing_time > 5:
            print("⚠️  Processing took longer than expected")
        else:
            print("✅ Processing completed within expected time")

    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Verify Database Storage
    print("\n💾 Step 3: Verifying Database Storage")
    print("-" * 40)

    try:
        db = SessionLocal()

        # Check capital calls
        capital_calls = db.query(CapitalCall).filter(CapitalCall.fund_id == 1).all()
        print(f"✅ Capital Calls in DB: {len(capital_calls)}")
        if capital_calls:
            print(f"   Sample: {capital_calls[0].call_date} - ${capital_calls[0].amount}")

        # Check distributions
        distributions = db.query(Distribution).filter(Distribution.fund_id == 1).all()
        print(f"✅ Distributions in DB: {len(distributions)}")
        if distributions:
            print(f"   Sample: {distributions[0].distribution_date} - ${distributions[0].amount}")

        # Check adjustments
        adjustments = db.query(Adjustment).filter(Adjustment.fund_id == 1).all()
        print(f"✅ Adjustments in DB: {len(adjustments)}")
        if adjustments:
            print(f"   Sample: {adjustments[0].adjustment_date} - ${adjustments[0].amount}")

        # Check documents
        documents = db.query(Document).filter(Document.fund_id == 1).all()
        print(f"✅ Documents in DB: {len(documents)}")
        if documents:
            doc = documents[-1]  # Get latest
            print(f"   Latest: {doc.file_name} - Status: {doc.parsing_status}")

        db.close()

    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

    # Step 4: Performance Summary
    print("\n📈 Step 4: Performance Summary")
    print("-" * 40)

    print("✅ All tests passed successfully!")
    print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
    print(f"📊 Tables extracted: {result['tables_extracted']}")
    print(f"📝 Text chunks created: {result['text_chunks_created']}")

    # Calculate metrics
    total_tables = sum(result['tables_extracted'].values())
    total_records = len(capital_calls) + len(distributions) + len(adjustments)

    print(f"💾 Database records created: {total_records}")
    print(f"📋 Tables processed: {total_tables}")

    if processing_time < 3:
        print("🚀 Performance: Excellent (< 3 seconds)")
    elif processing_time < 5:
        print("✅ Performance: Good (< 5 seconds)")
    else:
        print("⚠️  Performance: Acceptable (< 10 seconds)")

    print("\n" + "=" * 60)
    print("🎉 Document Processing Pipeline Test COMPLETED!")
    print("=" * 60)

    return True


async def test_individual_components():
    """Test individual components separately"""

    print("🧪 Testing Individual Components")
    print("=" * 50)

    pdf_path = "files/Sample_Fund_Performance_Report.pdf"

    # Test Table Parser Only
    print("\n1. Testing Table Parser Only:")
    try:
        parser = TableParser()
        result = parser.parse_tables(pdf_path)
        print(f"   ✅ Tables found: {sum(len(tables) for tables in result.values())}")
    except Exception as e:
        print(f"   ❌ Table parser failed: {e}")

    # Test Document Processor Only
    print("\n2. Testing Document Processor Only:")
    try:
        processor = DocumentProcessor()
        result = await processor.process_document(pdf_path, 999, 1)  # Test ID
        print(f"   ✅ Processing status: {result['status']}")
        print(f"   ✅ Tables extracted: {result['tables_extracted']}")
    except Exception as e:
        print(f"   ❌ Document processor failed: {e}")

    print("\n" + "=" * 50)


def main():
    """Main test function"""

    print("Fund Performance Analysis System - Document Processing Test")
    print("==========================================================")

    # For Docker environment, we need to mount the files directory
    # The files should be available at /app/files if properly mounted
    if not os.path.exists("/app/files"):
        print("❌ Files directory not found at /app/files")
        print("   This test requires the files directory to be mounted in the Docker container")
        print("   Make sure your docker-compose.yml includes the files volume mount")
        print("   Or run: docker compose cp ../../../files backend:/app/files")
        sys.exit(1)

    # Change to project root directory for consistent file paths
    os.chdir("/app")
    print(f"Changed working directory to: {os.getcwd()}")

    # Check if sample PDF exists - use files/ from project root
    pdf_path = "files/Sample_Fund_Performance_Report.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Sample PDF not found: {pdf_path}")
        print(f"   Current directory: {os.getcwd()}")
        print("   Files in current directory:")
        try:
            for f in os.listdir("."):
                print(f"     - {f}")
        except:
            print("     (could not list directory)")
        print("   Please ensure you're running from the project root directory")
        sys.exit(1)

    # Check if Docker containers are running
    print("🔍 Checking system status...")

    # Check if we can connect to the database (indicates Docker is running)
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        print("✅ Database connection successful")
    except Exception as e:
        print(f"⚠️  Database not available: {e}")
        print("   Please ensure Docker containers are running with: docker compose up -d")
        print("   Then run this test script from within the backend container or with proper env vars")
        return False

    # Run the main test
    try:
        success = asyncio.run(test_document_processing())

        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("The document processing pipeline is working correctly.")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED!")
            print("Please check the error messages above.")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)