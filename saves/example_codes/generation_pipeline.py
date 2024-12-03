if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='Input textual data on covered topics', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='Output json dataset with student-teacher interactions', type=argparse.FileType('w'))
    parser.add_argument('-num', required=False, help='Number of conversations to generate', type=int)
    args = parser.parse_args()

    # Attributes of socratic conversations
    depth = 5 # Depth of conversations
    # chunk_size = 1000 # Chunk size of splits in input file
    num_conversations = 5 # Number of conversations
    if args.num:
        num_conversations = args.num

    # Run pipeline
    pipeline(args.i, args.o, num_conversations)