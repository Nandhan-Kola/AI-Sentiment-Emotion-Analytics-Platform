from django import forms

class UploadCSVForm(forms.Form):
    file = forms.FileField(required=True)

class ClassifyForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea, required=True)
