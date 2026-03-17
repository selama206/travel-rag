from main import query, ingest, collection


def ask(question: str):
    """
    calls query() directly and prints the answer
    """
    result = query(question)
    print(f"\n{result['answer']}")
    print(f"\nsources used: {', '.join(result['sources'])}\n")


def main():
    """
    the entry point of the cli
    """
    print("=== Travel RAG ===")

    if collection.count() == 0:
        print("no itineraries loaded yet — ingesting now...")
        ingest()
    else:
        print("itineraries already loaded and ready\n")

    while True:
        try:
            question = input("ask a question (or type 'exit' to quit): ").strip()

            if not question:
                continue

            if question.lower() == "exit":
                print("bye!")
                break

            ask(question)

        except KeyboardInterrupt:
            print("\nbye!")
            break


if __name__ == "__main__":
    main()