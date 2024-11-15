import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelBinarizer

def read_labels(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def combine_labels(labels):
    # Extract relation type from each label
    relation_types = [label.split('\t')[1] for label in labels]
    return relation_types

def main():
    if len(sys.argv) != 3:
        print("Usage: python scorer.py <answers_path> <proposed_answers_path>")
        sys.exit(1)

    answers_path = sys.argv[2]
    proposed_answers_path = sys.argv[1]

    # Read labels from files
    true_labels = read_labels(answers_path)
    predicted_labels = read_labels(proposed_answers_path)

    # Combine labels to get relation types
    true_relation_types = combine_labels(true_labels)
    predicted_relation_types = combine_labels(predicted_labels)

    # Get all unique classes
    all_classes = sorted(set(true_relation_types + predicted_relation_types))

    # Convert relation types to binary format
    lb = LabelBinarizer()
    lb.fit(all_classes)  # Ensure all classes are considered
    true_bin = lb.transform(true_relation_types)
    pred_bin = lb.transform(predicted_relation_types)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_bin.argmax(axis=1), pred_bin.argmax(axis=1))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    classes = lb.classes_
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png", format="png")
    plt.show()

    # Calculate metrics
    accuracy = accuracy_score(true_bin, pred_bin)
    weighted_f1 = f1_score(true_bin, pred_bin, average='weighted')
    weighted_precision = precision_score(true_bin, pred_bin, average='weighted', zero_division=0)
    weighted_recall = recall_score(true_bin, pred_bin, average='weighted', zero_division=0)
    macro_f1 = f1_score(true_bin, pred_bin, average='macro')
    macro_precision = precision_score(true_bin, pred_bin, average='macro', zero_division=0)
    macro_recall = recall_score(true_bin, pred_bin, average='macro', zero_division=0)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_bin, pred_bin, target_names=lb.classes_))

    # Print results
    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nAccuracy:", accuracy)
    print("Weighted F1 Score:", weighted_f1)
    print("Weighted Precision:", weighted_precision)
    print("Weighted Recall:", weighted_recall)
    print("Macro F1 Score:", macro_f1)
    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)


if __name__ == "__main__":
    main()

