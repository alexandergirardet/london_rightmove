resource "aws_db_instance" "realestate-database" {
  # These fields are examples; modify them according to your existing resource's configuration
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "12.3"
  instance_class       = "db.t3.micro"
  username             = "postgres"
  password             = var.db_password
}
