import csv
import json
from datetime import datetime
from app.buffybot import BuffyBot


class RAGTester:
    def __init__(self):
        self.buffybot = BuffyBot()
        self.results = []

        # Define your test queries here
        self.test_queries = [
            # Character personality tests
            "What's your biggest fear?",
            "Tell me about your family",
            "How do you handle stress?",
            "What's your worst memory?",
            "What makes you happy?",

            # Buffy-specific scenarios
            "Who is the toughest Big Bad?",
            "Tell me about your relationship with your mom",
            "What's the hardest part about fighting the forces of evil?",
            "How do you feel about prophecies?",
            "What's your favorite weapon?",

            # Conversation starters (test for blandness)
            "Hey, how's your day going?",
            "What do you think about love?",
            "Tell me something interesting about yourself",
            "What's on your mind lately?",
            "Give me some advice",
        ]

        self.characters = ["Buffy", "Willow", "Xander", "Giles", "Spike"]

    def run_single_test(self, character, query):
        """Run a single query test and return results"""
        try:
            # Start fresh conversation for each test
            conversation = self.buffybot.start_conversation(character)
            session_id = conversation['session_id']

            # Send the query
            response = self.buffybot.chat(session_id, query)

            # Extract and clean the data
            result = {
                'timestamp': datetime.now().isoformat(),
                'character': character,
                'query': query,
                'response_text': response.get('response', ''),
                'confidence': response.get('confidence', 0),
                'chunks_used_count': len(response.get('chunks_used', [])),
                'conversation_length': response.get('conversation_length', 0),
                'session_id': response.get('session_id', ''),

                # Episode context info
                'source_episodes': [chunk.get('episode_title', '') for chunk in response.get('chunks_used', [])],
                'similarity_scores': [chunk.get('similarity_score', 0) for chunk in response.get('chunks_used', [])],
                'avg_similarity': sum(
                    [chunk.get('similarity_score', 0) for chunk in response.get('chunks_used', [])]) / max(
                    len(response.get('chunks_used', [])), 1),

                # Raw chunks for detailed analysis
                'retrieved_chunks': response.get('chunks_used', [])
            }

            return result

        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'character': character,
                'query': query,
                'error': str(e),
                'response_text': '',
                'confidence': 0
            }

    def run_full_test_suite(self):
        """Run all queries against all characters"""
        print("Starting RAG testing suite...")
        total_tests = len(self.characters) * len(self.test_queries)
        current_test = 0

        for character in self.characters:
            print(f"\nTesting {character}...")

            for query in self.test_queries:
                current_test += 1
                print(f"  [{current_test}/{total_tests}] {query[:50]}...")

                result = self.run_single_test(character, query)
                self.results.append(result)

        print(f"\nCompleted {len(self.results)} tests!")

    def save_results(self, filename=None):
        """Save results to CSV and JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_test_results_{timestamp}"

        # Save detailed results as JSON
        json_filename = f"{filename}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results saved to {json_filename}")

        # Save summary results as CSV for easy analysis
        csv_filename = f"{filename}.csv"
        csv_fields = [
            'timestamp', 'character', 'query', 'response_text',
            'confidence', 'chunks_used_count', 'avg_similarity',
            'source_episodes', 'error'
        ]

        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

            for result in self.results:
                # Flatten complex fields for CSV
                csv_row = {k: v for k, v in result.items() if k in csv_fields}
                csv_row['source_episodes'] = '; '.join(result.get('source_episodes', []))
                writer.writerow(csv_row)

        print(f"Summary results saved to {csv_filename}")

    def print_summary(self):
        """Print a quick summary of results"""
        if not self.results:
            print("No results to summarize")
            return

        print("\n=== TEST SUMMARY ===")
        print(f"Total tests: {len(self.results)}")

        # Character breakdown
        char_counts = {}
        for result in self.results:
            char = result.get('character', 'Unknown')
            char_counts[char] = char_counts.get(char, 0) + 1

        print("\nTests per character:")
        for char, count in char_counts.items():
            print(f"  {char}: {count}")

        # Confidence stats
        confidences = [r.get('confidence', 0) for r in self.results if 'error' not in r]
        if confidences:
            print(f"\nConfidence scores:")
            print(f"  Average: {sum(confidences) / len(confidences):.3f}")
            print(f"  Min: {min(confidences):.3f}")
            print(f"  Max: {max(confidences):.3f}")


def main():
    """Run the testing suite"""
    tester = RAGTester()

    # Run the tests
    tester.run_full_test_suite()

    # Save results
    tester.save_results()

    # Print summary
    tester.print_summary()

    print("\nTesting complete! Check the generated files for detailed analysis.")


if __name__ == "__main__":
    main()