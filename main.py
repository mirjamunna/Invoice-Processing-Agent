"""Main entry point for the Invoice Processing Agent."""

import sys

from invoice_agent.agent import create_invoice_agent

SAMPLE_INVOICE = """\
INVOICE

Invoice Number: INV-2024-001
Date: January 15, 2024

From:
  Acme Corp
  123 Business Ave, Suite 100
  Springfield, IL 62701

Bill To:
  Widget Industries
  456 Commerce St
  Shelbyville, IL 62565

Description                  Qty    Unit Price    Total
-----------------------------------------------------------
Web Development Services      40      $150.00    $6,000.00
UI/UX Design                  20      $125.00    $2,500.00
Server Hosting (Monthly)       1      $200.00      $200.00

                              Subtotal:          $8,700.00
                              Tax (8%):            $696.00
                              Total:             $9,396.00

Payment Terms: Net 30
Due Date: February 14, 2024
"""


def main():
    """Run the invoice processing agent with a sample invoice."""
    print("=" * 60)
    print("Invoice Processing Agent")
    print("=" * 60)

    agent = create_invoice_agent()

    if len(sys.argv) > 1:
        # Read invoice text from a file if provided
        with open(sys.argv[1]) as f:
            invoice_text = f.read()
    else:
        invoice_text = SAMPLE_INVOICE
        print("\nUsing sample invoice (pass a file path as argument for custom input).\n")

    prompt = f"Please process the following invoice and store it:\n\n{invoice_text}"
    print("Processing invoice...\n")

    response = agent.process(prompt)
    print(response)


if __name__ == "__main__":
    main()
