<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>KWIX project</title>
    <link rel="stylesheet" href="css/bootstrap.css">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-iYQeCzEYFbKjA/T2uDLTpkwGzCiq6soy8tYaI1GyVh/UjpbCx/TYkiZhlZB6+fzT" crossorigin="anonymous">
</head>
<body class="text-bg-dark">
    <div class="container">

<?php
  $host = 'localhost';
  $username = 'root';
  $password = '12345678';
  $dbname = 'testdb';

  $id = $_POST["id"];
  $name = $_POST["name"];
  $location = $_POST["location"];
  $concentration = $_POST["concentration"];
  $result = $_POST["result"];

  $conn = mysqli_connect($host, $username, $password, $dbname);

  $updateQuery = "UPDATE local_num SET num = num + 1 WHERE location = '{$location}';";
  $result2 = mysqli_query($conn,$updateQuery);

  $res = mysqli_query($conn, "SELECT num FROM local_num WHERE location = '{$location}'");
  $row = mysqli_fetch_array($res);

  $insertQuery = "INSERT INTO covid VALUES ('{$id}','{$name}','{$location}','".date("Y-m-d h:i:sa")."','{$concentration}',{$result},{$row['num']});";
  $result = mysqli_query($conn,$insertQuery);
  
  if($result){
    echo "<h1>데이터 입력 완료</h1><br>";
    echo "<h5>id: ".$id."</h5><br><h5>name: ".$name."</h5><br><h5>location: ".$location."</h5><br><h5>concentration: ".
    $concentration."</h5><br><h5>result: ".$result."</h5><br>";} else{echo "<h1>데이터 입력 실패</h1>";}
    ?>
  <button onClick="location.href='/index.php'" type="button" class="btn-secondary fw-bold">돌아가기</button>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-A3rJD856KowSb7dwlZdYEkO39Gagi7vIsF0jrRAoQmDKKtQBHUuLZ9AsSv4jD4Xa" crossorigin="anonymous"></script>
</body>
</html>