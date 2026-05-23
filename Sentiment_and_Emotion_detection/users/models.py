from django.db import models

class UploadedDataset(models.Model):
    uploaded_at = models.DateTimeField(auto_now_add=True)
    original_filename = models.CharField(max_length=255)
    stored_path = models.CharField(max_length=1024)
    row_count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.original_filename} ({self.row_count} rows)"
