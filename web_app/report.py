from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import time
import os

def generate_pdf(score_history, feedback_list):

    file_name = f"shadow_report_{int(time.time())}.pdf"

    c = canvas.Canvas(file_name, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "SHADOW SKILL TRAINING REPORT")

    c.setFont("Helvetica", 12)

    # Average score
    avg_score = sum(score_history) / len(score_history) if score_history else 0

    c.drawString(50, height - 100, f"Average Score: {avg_score:.2f}%")
    c.drawString(50, height - 130, f"Total Frames: {len(score_history)}")

    # Feedback summary
    c.drawString(50, height - 170, "Feedback Summary:")

    y = height - 200
    for fb in feedback_list[-10:]:
        c.drawString(60, y, f"- {fb}")
        y -= 20

        if y < 100:
            c.showPage()
            y = height - 50

    c.save()

    return file_name