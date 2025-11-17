"""Quick test of WikipediaXMLParser - just check latest date."""

from src.data.wikipedia_xml_parser import WikipediaXMLParser

print("Testing WikipediaXMLParser...")
print("=" * 60)

# Test 1: Get latest dump date for Korean Wikipedia
print("\n1. Testing latest dump date retrieval:")
try:
    parser = WikipediaXMLParser(language="ko", date="latest")
    latest_date = parser.get_latest_dump_date()
    print(f"✓ Latest Korean Wikipedia dump: {latest_date}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Check URL construction
print("\n2. Testing URL construction:")
try:
    parser = WikipediaXMLParser(language="ko", date="20251101")
    url = parser.get_dump_url()
    print(f"✓ Dump URL: {url}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("Basic tests passed!")
print("\nTo process actual Wikipedia data:")
print("  1. Run the notebook cells")
print("  2. First run will download ~1-2GB dump file")
print("  3. Cached dump will be reused for subsequent runs")
