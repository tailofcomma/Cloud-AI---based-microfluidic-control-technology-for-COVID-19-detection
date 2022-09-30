<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>KWIX project</title>
    <link rel="stylesheet" href="css/bootstrap.css">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-iYQeCzEYFbKjA/T2uDLTpkwGzCiq6soy8tYaI1GyVh/UjpbCx/TYkiZhlZB6+fzT" crossorigin="anonymous">
		<link rel="stylesheet" href="assets/css/korea-map-font-v1.css">

		<style>
			.korea-map-font-v1 {
  			width: 1000px;
  			font-size: 800px;
  			color: gray;
			}
			.서울{
  			
        
			}
      .jumbotron{
        background-size: cover;
        text-shadow: black 0.2em 0.2em 0.2em;
        color: white;
        
      }
      
		</style>

  </head>
  <body class="text-bg-dark">
    <div class="container">
      <div class="jumbotron">
        <h1 class="text-center">COVID19 Diagnosis system Map</h1>
    
        <p class="text-center">(일일 확진자수 기준)</p>
      
        <p class="text-center lead"><button onClick="window.location.reload()" type="button" class="btn-secondary fw-bold">로드하기</button></p>
        
        <p class="text-center lead"><button onClick="location.href='Qr.html'" type="button" class="btn-secondary fw-bold">QR-Code</button></p>
        <br>
      </div>
    </div>
    
    <div class="row">
      <div class="col-lg-4">

        <h1>Korea map</h1>
        <p>(지역에 따른 확진자수 분포)</p>
        
        
    <span class="badge bg-success" style="background-color:green">~10</span>
    <span class="badge bg-warning text-dark" style="background-color:yellow">10~50</span>
    <span class="badge bg-danger" style="background-color:red">50~</span>
<ul class="korea-map-font-v1">
  <li class="강원">a</li>
  <li class="경기">b</li>
  <li class="경남">c</li>
  <li class="경북">d</li>
  <li class="광주">e</li>
  <li class="대구">f</li>
  <li class="대전">g</li>
  <li class="부산">h</li>
  <li class="서울">i</li>
  <li class="세종">j</li>
  <li class="울산">k</li>
  <li class="인천">l</li>
  <li class="전남">m</li>
  <li class="전북">n</li>
  <li class="제주">o</li>
  <li class="충남">p</li>
  <li class="충북">q</li>
</ul>
    </div>
    <div class="col-lg-3">
    </div>
    <div class="col-lg-4" style="overflow:auto; height:800px">
        <h2>Confirmed Table</h2>
        
          <table class="table table-sm" style="color:white;">
            <thead style="color:white">
              <tr>
                <th scope="col">ID</th>
                <th scope="col">NAME</th>
                <th scope="col">LOCATION</th>
                <th scope="col">DATETIME</th>
                <th scope="col">CONCENTRATION</th>
                <th scope="col">RESULT</th>
                <th scope="col">LOCAL_RESULT</th>
              </tr>
            </thead>
            <tbody style="color:white">
            <?php

            $host = 'localhost';
            $username = 'root'; # MySQL 계정 아이디
            $password = '12345678'; # MySQL 계정 패스워드
            $dbname = 'testdb';  # DATABASE 이름

            $conn = mysqli_connect($host, $username, $password, $dbname);
            $res = mysqli_query($conn, "SELECT * FROM covid ORDER BY datetime DESC");
            $res2 = mysqli_query($conn, "SELECT * FROM local_num");

            while($row = mysqli_fetch_array($res2)) {
              if($row['num'] < 10) {
                echo "<script>document.getElementsByClassName('{$row['location']}')[0].style.color = 'green';</script>";
              } else if(($row['num'] < 50) && ($row['num'] >= 10)) {
                echo "<script>document.getElementsByClassName('{$row['location']}')[0].style.color = 'yellow';</script>";

              } else if(($row['num'] >= 50)) {

                echo "<script>document.getElementsByClassName('{$row['location']}')[0].style.color = 'red';</script>"; }

            }
           # if($res == false){
            #echo "저장하는 과정에서 문제가 생겼습니다.";
            #error_log(mysqli_error($conn));
            #} else {
            while($ROW = mysqli_fetch_array($res)) {

              echo "<tr>";
              echo "<td>{$ROW['id']}</td><td>{$ROW['name']}</td><td>{$ROW['location']}</td><td>{$ROW['datetime']}</td><td>{$ROW['concentration']}</td><td>{$ROW['result']}</td><td>{$ROW['local_result']}</td></tr>";
              
            }# }
?>
            </tbody>
          </table>
      </div>
      <div class="col-lg-1">
      </div>
      
    </div>
  <footer class="mt-auto text-white-50">
    <div class="container">
      <div class='row'>
        <h5 class="text-center">Copyright &copy; 2022 KWIX team(Bio-Laboratory, 02-940-8671)</h5>
      </div>  
      <div class="row">
        <p class="text-center">
          이 홈페이지는 미세유체 제어기술을 활용한 코로나19 검출 시스템에 활용되고 있습니다.
        </p>
      </div>
    </div>

  </footer>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-A3rJD856KowSb7dwlZdYEkO39Gagi7vIsF0jrRAoQmDKKtQBHUuLZ9AsSv4jD4Xa" crossorigin="anonymous"></script>
  </body>
</html>