
# # Prediction & Evaluation
# y_pred = model.predict(x_test).reshape(len(x_test))
# threshold = 0.1
# y_pred_threshold = [1 if value > threshold else 0 for value in y_pred]
# confMatrix = confusion_matrix(y_test, y_pred_threshold)
# print("Confusion matrix with following shape:\n"
#       "[[TN FP]\n"
#       " [FN TP]]")
# print(confusion_matrix(y_test, y_pred_threshold))
# print("Sensitivity/Recall=" + str(recall_score(y_test, y_pred_threshold)))
# specificity = float(confMatrix[0][0]) / (
#     confMatrix[0][0] + confMatrix[0][1])  # Couldn't find specificity in Scikit-learn
# print("Specificity=" + str(specificity))
# print("MCC=" + str(matthews_corrcoef(y_test, y_pred_threshold)))
